from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_):  # type: ignore[misc]
        return it


@dataclass
class TrainingConfig:
    epochs: int = 10
    effective_batch_size: int = 2048
    per_device_batch_size: int = 32
    learning_rate: float = 7e-4
    lr_scheduler: str = "cosine"
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    max_seq_len: int = 256
    fp16: bool = False


def train_surrogate(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
) -> list[str]:
    torch.manual_seed(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    # Pre-pad to fixed length → TensorDataset with no per-batch collation overhead
    batch_out = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = batch_out["input_ids"]       # (N, L)
    attn_mask = batch_out["attention_mask"]  # (N, L)
    dataset = TensorDataset(input_ids, attn_mask)

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    grad_accum = max(1, config.effective_batch_size // config.per_device_batch_size)
    steps_per_epoch = max(1, math.ceil(len(dataset) / config.per_device_batch_size))
    total_steps = max(1, config.epochs * math.ceil(steps_per_epoch / grad_accum))
    scheduler = get_scheduler(
        config.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=total_steps,
    )

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []

    pin = device == "cuda"
    indices = list(range(len(dataset)))
    rng = random.Random(seed)

    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    pbar = tqdm(total=total_steps, desc="surrogate", unit="step")

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        loader = DataLoader(
            dataset,
            batch_size=config.per_device_batch_size,
            sampler=indices,
            pin_memory=pin,
        )
        optimizer.zero_grad()
        running_loss, accum_count = 0.0, 0

        for step, (ids, mask) in enumerate(loader):
            ids  = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = ids.clone()
            labels[mask == 0] = -100   # ignore padding in loss
            with torch.cuda.amp.autocast(enabled=config.fp16):
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            scaler.scale(outputs.loss / grad_accum).backward()
            running_loss += outputs.loss.item()
            accum_count += 1

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(epoch=epoch, loss=f"{running_loss/accum_count:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                running_loss, accum_count = 0.0, 0

        if len(loader) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

        path = str(ckpt_dir / f"epoch_{epoch:02d}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        checkpoint_paths.append(path)
        tqdm.write(f"epoch {epoch:02d} done — checkpoint saved to {path}")

    pbar.close()
    model.eval()
    optimizer.zero_grad()
    return checkpoint_paths
