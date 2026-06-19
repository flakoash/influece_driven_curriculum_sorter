from __future__ import annotations

import gc
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

try:
    from torch.func import functional_call, jvp
    _HAS_JVP = True
except ImportError:
    _HAS_JVP = False

# Legacy vmap flag kept for test compatibility
try:
    from torch.func import grad as fgrad, vmap
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
    grad_batch_size: int = 4    # batch size for pass-2 JVP (safe on T4; increase for speed)
    fp16: bool = True           # load influence model in fp16 to halve GPU memory


def _unit(g: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(g)
    return g / n if n > 1e-10 else g


def _unit_batch(t: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization for a 2-D tensor (B, D)."""
    norms = t.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return t / norms


# ── kept for test_correctness.py ─────────────────────────────────────────────

def _vmap_grads(model, emb_name, batch_ids, batch_masks):
    """Per-sample embedding gradients via vmap (test helper only — see _jvp_score)."""
    emb_weight = dict(model.named_parameters())[emb_name]
    other_params = {n: p for n, p in model.named_parameters() if n != emb_name}
    buffers = dict(model.named_buffers())

    def loss_fn(emb_w, ids, mask):
        labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
        all_params = {**other_params, emb_name: emb_w}
        out = functional_call(
            model, (all_params, buffers), args=(),
            kwargs={"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0),
                    "labels": labels.unsqueeze(0)},
        )
        return out.loss

    per_sample = vmap(fgrad(loss_fn, argnums=0), in_dims=(None, 0, 0))
    emb_grad = per_sample(emb_weight, batch_ids, batch_masks)
    return emb_grad.reshape(len(batch_ids), -1)


# ── production helpers ────────────────────────────────────────────────────────

def _fallback_grad(model, ids, mask, device):
    """Per-doc gradient without vmap (CPU fallback)."""
    ids  = (ids  if ids.dim()  == 2 else ids.unsqueeze(0)).to(device)
    mask = (mask if mask.dim() == 2 else mask.unsqueeze(0)).to(device)
    labels = ids.clone()
    labels[mask == 0] = -100
    emb = model.get_input_embeddings()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    (g,) = torch.autograd.grad(out.loss, emb.weight)
    return g.detach().cpu().float().numpy().ravel().copy()


def _jvp_score(model, emb_name, other_params, buffers, emb_w, ids, mask, mean_g):
    """Directional derivative ∂loss/∂emb_w · mean_g for one doc via JVP.

    Forward-mode AD: no backward graph, no create_graph=True, no memory accumulation.
    """
    def f_emb(w):
        labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
        all_p = {**other_params, emb_name: w}
        return functional_call(
            model, (all_p, buffers), args=(),
            kwargs={"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0),
                    "labels": labels.unsqueeze(0)},
        ).loss

    _, tangent = jvp(f_emb, (emb_w,), (mean_g.reshape(emb_w.shape),))
    return tangent.item()


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
    return torch.stack(pad_ids), torch.stack(pad_masks)


# ── main entry point ──────────────────────────────────────────────────────────

def compute_influence_matrix(
    checkpoint_paths: list[str],
    encodings: list[dict],
    config: InfluenceConfig,
    device: str,
) -> np.ndarray:
    """Compute Phi[i,t] = influence of doc i at checkpoint t.

    Algorithm:
      pass 1 – mean embedding gradient via mini-batch backward (B=256, fast)
      pass 2 – per-doc score via JVP (forward-mode AD, no backward graph → no OOM)
    """
    D = len(encodings)
    if D == 0:
        raise ValueError("encodings list is empty — no documents to score")
    T = len(checkpoint_paths)
    Phi = np.zeros((D, T), dtype=np.float32)

    all_ids, all_masks = _pad_encodings(encodings)   # (D, L) on CPU

    use_jvp = _HAS_JVP and device != "cpu"

    if device.startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()

    for t, ckpt in enumerate(checkpoint_paths):
        model = AutoModelForCausalLM.from_pretrained(ckpt, attn_implementation="sdpa")
        # fp16 breaks JVP tangent propagation (dtype mismatch at LayerNorm/GELU).
        # Per-doc JVP peak memory is tiny (~200 MB), so fp32 is fine.
        if config.fp16 and device != "cpu" and not use_jvp:
            model = model.half()
        model = model.to(device).eval()

        emb = model.get_input_embeddings()
        emb_name = next(n for n, p in model.named_parameters() if p is emb.weight)
        grad_dim = emb.weight.numel()

        # ── Pass 1: mean gradient via mini-batch backward ─────────────────────
        # Only the embedding weight needs gradients — saves memory during backward.
        for p in model.parameters():
            p.requires_grad_(p is emb.weight)

        B1 = 256   # large batch; standard backward, no per-sample graph
        n_b1 = math.ceil(D / B1)
        mean_g = torch.zeros(grad_dim, dtype=torch.float32)

        for start in tqdm(range(0, D, B1), desc=f"ckpt {t} pass1", total=n_b1, leave=False):
            ids  = all_ids[start:start + B1].to(device)
            mask = all_masks[start:start + B1].to(device)
            n_doc = ids.shape[0]
            labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
            # multiply by n_doc so we can sum→mean correctly across batches
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss * n_doc
            loss.backward()
            mean_g += emb.weight.grad.detach().cpu().float().ravel() * n_doc
            emb.weight.grad = None
            del ids, mask, labels, loss

        mean_g /= D
        if config.normalize:
            n = mean_g.norm().item()
            if n > 1e-10:
                mean_g = mean_g / n

        # ── Pass 2: per-doc scores ────────────────────────────────────────────
        if use_jvp:
            # Forward-mode AD: directional derivative ∂loss/∂emb · mean_g.
            # JVP never calls create_graph=True — no memory accumulation across docs.
            other_params = {n: p.detach() for n, p in model.named_parameters() if n != emb_name}
            buffers = dict(model.named_buffers())
            emb_w = emb.weight.detach()   # fp32 when use_jvp=True
            mean_g_dev = mean_g.to(device)

            for i in tqdm(range(D), desc=f"ckpt {t} pass2", leave=False):
                ids  = all_ids[i:i + 1].to(device)
                mask = all_masks[i:i + 1].to(device)
                Phi[i, t] = _jvp_score(model, emb_name, other_params, buffers,
                                        emb_w, ids[0], mask[0], mean_g_dev)
                del ids, mask

        else:
            # CPU fallback: per-doc backward (no torch.func required)
            mean_g_np = mean_g.numpy()
            for i in tqdm(range(D), desc=f"ckpt {t} pass2 (cpu)", leave=False):
                g = _fallback_grad(model, all_ids[i], all_masks[i], device)
                if config.normalize:
                    g = _unit(g)
                Phi[i, t] = float(np.dot(g, mean_g_np))

        del model, mean_g
        if device.startswith("cuda"):
            gc.collect()
            torch.cuda.empty_cache()

    return Phi
