from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM

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
    grad_batch_size: int = 16


def _doc_gradient(model: torch.nn.Module, input_ids: torch.Tensor, device: str) -> np.ndarray:
    ids = (input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)).to(device)
    emb = model.get_input_embeddings()
    outputs = model(input_ids=ids, labels=ids)
    # autograd.grad targets only the embedding weight — avoids zero_grad overhead
    # and skips allocating .grad tensors for all other parameters
    (grad,) = torch.autograd.grad(outputs.loss, emb.weight)
    return grad.detach().cpu().float().numpy().ravel().copy()


def _unit(g: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(g)
    return g / n if n > 1e-10 else g


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

    for t, ckpt in enumerate(checkpoint_paths):
        model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
        model.eval()

        # Require grad only on embedding — backward skips accumulation for all other params
        emb_weight = model.get_input_embeddings().weight
        for p in model.parameters():
            p.requires_grad_(p is emb_weight)

        grad_dim = emb_weight.numel()

        # Pass 1: mean gradient
        mean_g = np.zeros(grad_dim, dtype=np.float64)
        for enc in tqdm(encodings, desc=f"ckpt {t} pass1", leave=False):
            g = _doc_gradient(model, enc["input_ids"], device)
            if config.normalize:
                g = _unit(g)
            mean_g += g
        mean_g /= D

        # Pass 2: per-doc score
        for i, enc in enumerate(tqdm(encodings, desc=f"ckpt {t} pass2", leave=False)):
            g = _doc_gradient(model, enc["input_ids"], device)
            if config.normalize:
                g = _unit(g)
            Phi[i, t] = float(np.dot(g, mean_g))

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return Phi
