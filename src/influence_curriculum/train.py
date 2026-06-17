from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler


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


class _TokenDataset(Dataset):
    def __init__(self, encodings: list[dict]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, i: int) -> dict:
        return {k: v.squeeze(0) for k, v in self.encodings[i].items()}


def _collate(batch: list[dict], tokenizer) -> dict:
    pad_values = {
        "input_ids": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        "labels": -100,
        "attention_mask": 0,
    }
    keys = batch[0].keys()
    return {k: torch.nn.utils.rnn.pad_sequence(
        [b[k] for b in batch], batch_first=True, padding_value=pad_values.get(k, 0)
    ) for k in keys}


def train_surrogate(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    output_dir: str,
    config: TrainingConfig,
    seed: int,
    device: str,
) -> list[str]:
    # Re-initialize all weights from scratch (spec: architecture only, not weights)
    torch.manual_seed(seed)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    encodings = [
        tokenizer(t, truncation=True, max_length=config.max_seq_len, return_tensors="pt")
        for t in texts
    ]
    dataset = _TokenDataset(encodings)

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

    indices = list(range(len(dataset)))
    rng = random.Random(seed)

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        loader = DataLoader(
            dataset, batch_size=config.per_device_batch_size,
            sampler=indices, collate_fn=lambda batch: _collate(batch, tokenizer),
        )
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**model_inputs, labels=labels)
            (outputs.loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # flush any remaining accumulated gradients
        if len(loader) % grad_accum != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        path = str(ckpt_dir / f"epoch_{epoch:02d}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        checkpoint_paths.append(path)

    model.eval()
    optimizer.zero_grad()
    return checkpoint_paths
