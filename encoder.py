# encoder.py
# RAE-HMC M1: Shared encoder for inputs and label descriptions.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

Tensor = torch.Tensor


@dataclass
class EncoderConfig:
    model_name: str = "bert-base-chinese"
    max_length: int = 32
    pooling: str = "mean"  # {"cls", "mean"}
    normalize: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False
    amp_enabled: bool = False
    amp_dtype: str = "bf16"
    grad_checkpointing: bool = False


def _batchify(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _mean_pool(last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


class SharedEncoder(nn.Module):
    """
    One transformer encoder shared by sample text and label text.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        if cfg.grad_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.to(cfg.device)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _pool(self, last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
        if self.cfg.pooling.lower() == "mean":
            pooled = _mean_pool(last_hidden, attention_mask)
        else:
            pooled = last_hidden[:, 0, :]
        if self.cfg.normalize:
            pooled = _l2_normalize(pooled)
        return pooled

    def _autocast_context(self):
        device = str(self.cfg.device)
        amp_enabled = bool(getattr(self.cfg, "amp_enabled", False) or getattr(self.cfg, "fp16", False))
        if not amp_enabled or not device.startswith("cuda"):
            return nullcontext()
        amp_dtype = str(getattr(self.cfg, "amp_dtype", "bf16")).strip().lower()
        dtype = torch.float16 if amp_dtype in {"fp16", "float16", "half"} else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        progress: bool = False,
        **token_kwargs,
    ) -> Tensor:
        device = self.cfg.device
        outputs: List[Tensor] = []

        iterable = _batchify(texts, batch_size)
        if progress:
            try:
                from tqdm import tqdm

                iterable = tqdm(
                    iterable,
                    total=(len(texts) + batch_size - 1) // batch_size,
                    desc="Encoding",
                )
            except Exception:
                pass

        autocast_ctx = self._autocast_context
        self.eval()
        with torch.no_grad():
            for batch_texts in iterable:
                batch = self.tokenizer(
                    batch_texts,
                    max_length=self.cfg.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    **token_kwargs,
                ).to(device)
                with autocast_ctx():
                    out = self.model(**batch)
                    pooled = self._pool(out.last_hidden_state, batch["attention_mask"])
                outputs.append(pooled.detach().cpu())

        return torch.cat(outputs, dim=0)

    def encode_texts(self, texts: List[str], batch_size: int = 64, progress: bool = False) -> Tensor:
        return self._encode_batch(texts, batch_size=batch_size, progress=progress)

    def encode_labels(self, label_descriptions: List[str], batch_size: int = 128, progress: bool = False) -> Tensor:
        return self._encode_batch(label_descriptions, batch_size=batch_size, progress=progress)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Optional[Tensor] = None) -> Tensor:
        with self._autocast_context():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled = self._pool(out.last_hidden_state, attention_mask)
        return pooled
