from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from torch.func import functional_call, grad as fgrad, vmap
    _HAS_VMAP = True
except ImportError:
    _HAS_VMAP = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):  # type: ignore[misc]
        return it


@dataclass
class InfluenceConfig:
    grad_target: str = "input_embeddings"
    grad_path: str = "full"
    normalize: bool = True
    memory_route: str = "recompute"
    projection_dim: int = 0
    grad_batch_size: int = 16   # docs per vmap batch; reduce if OOM


def _unit(g: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(g)
    return g / n if n > 1e-10 else g


def _unit_batch(t: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization for a 2-D tensor (B, D)."""
    norms = t.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return t / norms


def _vmap_grads(model, params, buffers, emb_name, batch_ids, batch_masks):
    """Per-sample embedding gradients via vmap. Shape: (B, vocab*hidden)."""
    def loss_fn(params, buffers, ids, mask):
        # torch.where is vmap-compatible; boolean mask indexing is not
        labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
        out = functional_call(
            model, (params, buffers), args=(),
            kwargs={"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0), "labels": labels.unsqueeze(0)},
        )
        return out.loss

    per_sample = vmap(fgrad(loss_fn), in_dims=(None, None, 0, 0))
    grads_dict = per_sample(params, buffers, batch_ids, batch_masks)
    return grads_dict[emb_name].reshape(len(batch_ids), -1)   # (B, grad_dim)


def _fallback_grad(model, ids, mask, device):
    """Per-doc gradient without vmap."""
    ids  = (ids  if ids.dim()  == 2 else ids.unsqueeze(0)).to(device)
    mask = (mask if mask.dim() == 2 else mask.unsqueeze(0)).to(device)
    labels = ids.clone()
    labels[mask == 0] = -100
    emb = model.get_input_embeddings()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    (g,) = torch.autograd.grad(out.loss, emb.weight)
    return g.detach().cpu().float().numpy().ravel().copy()


def _pad_encodings(encodings: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length encodings to uniform length. Returns (all_ids, all_masks)."""
    max_len = max(enc["input_ids"].shape[-1] for enc in encodings)
    pad_ids, pad_masks = [], []
    for enc in encodings:
        ids  = enc["input_ids"].squeeze(0)
        mask = enc.get("attention_mask", torch.ones_like(ids)).squeeze(0)
        pad_len = max_len - ids.shape[-1]
        if pad_len > 0:
            ids  = F.pad(ids,  (0, pad_len), value=0)
            mask = F.pad(mask, (0, pad_len), value=0)
        pad_ids.append(ids)
        pad_masks.append(mask)
    return torch.stack(pad_ids), torch.stack(pad_masks)   # (D, max_len) each


def compute_influence_matrix(
    checkpoint_paths: list[str],
    encodings: list[dict],
    config: InfluenceConfig,
    device: str,
) -> np.ndarray:
    D = len(encodings)
    if D == 0:
        raise ValueError("encodings list is empty — no documents to score")
    T = len(checkpoint_paths)
    Phi = np.zeros((D, T), dtype=np.float32)
    B = config.grad_batch_size

    all_ids, all_masks = _pad_encodings(encodings)   # (D, L) — handles variable lengths

    use_vmap = _HAS_VMAP and device != "cpu"

    for t, ckpt in enumerate(checkpoint_paths):
        # eager attention avoids SDPA's .item() calls which are vmap-incompatible
        attn_impl = "eager" if use_vmap else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(ckpt, attn_implementation=attn_impl).to(device)
        model.eval()

        emb = model.get_input_embeddings()
        grad_dim = emb.weight.numel()

        if use_vmap:
            emb_name = next(n for n, p in model.named_parameters() if p is emb.weight)
            params  = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            n_batches = math.ceil(D / B)

            # Pass 1: mean gradient (batched)
            mean_g = torch.zeros(grad_dim, dtype=torch.float64)
            for start in tqdm(range(0, D, B), desc=f"ckpt {t} pass1", total=n_batches, leave=False):
                batch_ids   = all_ids[start:start + B].to(device)
                batch_masks = all_masks[start:start + B].to(device)
                g = _vmap_grads(model, params, buffers, emb_name, batch_ids, batch_masks).double().cpu()
                if config.normalize:
                    g = _unit_batch(g.float()).double()
                mean_g += g.sum(dim=0)
            mean_g = (mean_g / D).float()

            # Pass 2: per-doc scores (batched dot product)
            for start in tqdm(range(0, D, B), desc=f"ckpt {t} pass2", total=n_batches, leave=False):
                batch_ids   = all_ids[start:start + B].to(device)
                batch_masks = all_masks[start:start + B].to(device)
                actual = batch_ids.shape[0]
                g = _vmap_grads(model, params, buffers, emb_name, batch_ids, batch_masks).float().cpu()
                if config.normalize:
                    g = _unit_batch(g)
                Phi[start:start + actual, t] = (g @ mean_g).numpy()

        else:
            # Fallback: per-doc loop
            emb.weight.requires_grad_(True)
            for p in model.parameters():
                if p is not emb.weight:
                    p.requires_grad_(False)

            mean_g_np = np.zeros(grad_dim, dtype=np.float64)
            for i in tqdm(range(D), desc=f"ckpt {t} pass1", leave=False):
                g = _fallback_grad(model, all_ids[i], all_masks[i], device)
                if config.normalize:
                    g = _unit(g)
                mean_g_np += g
            mean_g_np /= D

            for i in tqdm(range(D), desc=f"ckpt {t} pass2", leave=False):
                g = _fallback_grad(model, all_ids[i], all_masks[i], device)
                if config.normalize:
                    g = _unit(g)
                Phi[i, t] = float(np.dot(g, mean_g_np))

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return Phi
