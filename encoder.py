# encoder.py
# RAE-HMC M1: Shared Encoder (aligned with §3.3 of the thesis)
# - One shared Transformer encoder for both inputs x and label descriptions t_y
# - Supports CLS/mean pooling and optional L2 normalization
# - Provides a hierarchical contrastive loss (HCL) skeleton for §3.3 pretraining

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from contextlib import nullcontext

Tensor = torch.Tensor


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class EncoderConfig:
    model_name: str = "bert-base-chinese"   # §3.3: BERT-base / RoBERTa-base
    max_length: int = 32                    # §3.3: title-like short text
    pooling: str = "mean"                    # {"cls", "mean"}
    normalize: bool = True                  # cosine-friendly
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False                      # enable autocast for speed if you like
    grad_checkpointing: bool = False        # reduce memory if needed


# -----------------------------
# Utilities
# -----------------------------
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
    # last_hidden: [B, T, D], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B, 1]
    return summed / denom


def _l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


# -----------------------------
# Shared Encoder
# -----------------------------
class SharedEncoder(nn.Module):
    """
    A single transformer encoder shared for inputs x and label descriptions t_y.
    Outputs fixed-size embeddings ready for retrieval (§3.4) and classification (§3.5).

    Methods:
        encode_texts:  encode a list of raw texts (for X)
        encode_labels: encode a list of label descriptions (for Z)
        forward:       encode a pre-tokenized batch (training)
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
            # CLS pooling: take [CLS] = token 0
            pooled = last_hidden[:, 0, :]
        if self.cfg.normalize:
            pooled = _l2_normalize(pooled)
        return pooled

    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        progress: bool = False,
        **token_kwargs
    ) -> Tensor:
        """
        Encode an arbitrary list of strings into a single tensor [N, D].
        """
        device = self.cfg.device
        outputs: List[Tensor] = []

        iterable = _batchify(texts, batch_size)
        if progress:
            try:
                from tqdm import tqdm  # optional dependency
                iterable = tqdm(iterable, total=(len(texts) + batch_size - 1) // batch_size, desc="Encoding")
            except Exception:
                pass

        autocast_ctx = torch.cuda.amp.autocast if (self.cfg.fp16 and device.startswith("cuda")) else nullcontext
        self.eval()
        with torch.no_grad():
            for batch_texts in iterable:
                batch = self.tokenizer(
                    batch_texts,
                    max_length=self.cfg.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    **token_kwargs
                ).to(device)

                with autocast_ctx():
                    out = self.model(**batch)
                    pooled = self._pool(out.last_hidden_state, batch["attention_mask"])
                outputs.append(pooled.detach().cpu())

        return torch.cat(outputs, dim=0)

    # Public APIs -------------------------------------------------------------

    def encode_texts(self, texts: List[str], batch_size: int = 64, progress: bool = False) -> Tensor:
        """
        Encode input texts x -> X  (for training set: X ∈ ℝ^{N×d})
        """
        return self._encode_batch(texts, batch_size=batch_size, progress=progress)

    def encode_labels(self, label_descriptions: List[str], batch_size: int = 128, progress: bool = False) -> Tensor:
        """
        Encode label descriptions t_y -> Z  (for label set: Z ∈ ℝ^{L×d})
        The description can be just the node name or a full path string (e.g., "食品 > 零食 > 巧克力").
        """
        return self._encode_batch(label_descriptions, batch_size=batch_size, progress=progress)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Optional[Tensor] = None) -> Tensor:
        """
        Forward for training (pre-tokenized). Returns pooled embeddings [B, d].
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self._pool(out.last_hidden_state, attention_mask)
    
    def encode_labels_from_hierarchy(self, hd, batch_size: int = 128, progress: bool = False):
        """
        依 HierarchyData 的全域 id 次序 (0..L-1) 取用 path_strings，產生 Z。
        確保 Z 的列序與 label2id 對齊。
        """
        descs = [hd.path_strings[i] for i in range(hd.num_labels)]
        return self.encode_labels(descs, batch_size=batch_size, progress=progress)


# -----------------------------
# Hierarchical Contrastive Loss (optional, §3.3)
# -----------------------------
class HierarchicalContrastiveLoss(nn.Module):
    """
    InfoNCE-style loss on label embeddings to enforce hierarchical geometry (§3.3).
    Positive pairs are (child -> parent) edges; negatives can be provided by label-specific pools.
    Inputs:
        Z:        [L, d] label embedding matrix (L2-normalized recommended)
        edges:    list of (p_idx, c_idx) for parent→child
        same_level_map (optional): dict[level] -> List[label_indices] to sample same-level negatives
        temperature: τ_h
        num_neg:  negatives per positive (if None, use all from candidate set)
    """

    def __init__(self, temperature: float = 0.07, num_neg: Optional[int] = 32):
        super().__init__()
        self.t = temperature
        self.num_neg = num_neg

    def forward(
        self,
        Z: Tensor,
        edges: List[Tuple[int, int]],
        same_level_map: Optional[Dict[int, List[int]]] = None,
        label_levels: Optional[List[int]] = None,
        num_neg: Optional[int] = None,
        neg_candidates_by_label: Optional[object] = None,
    ) -> Tensor:
        """
        Args:
            Z: [L, d]
        edges: list of (p, c) for parent/child
        same_level_map: {level: [indices]} to sample negatives from same level only
        label_levels: list length L with level index for each label
        neg_candidates_by_label: optional per-label negative pools (prefer over same_level_map)
        """
        device = Z.device
        if Z.ndim != 2:
            raise ValueError("Z must be 2D [L, d]")
        L = Z.size(0)

        sims = Z @ Z.t()  # [L, L], cosine if Z normalized
        loss_terms: List[Tensor] = []

        neg_pool = None
        neg_lengths = None
        if isinstance(neg_candidates_by_label, tuple) and len(neg_candidates_by_label) == 2:
            neg_pool, neg_lengths = neg_candidates_by_label
            if neg_pool.device != device:
                neg_pool = neg_pool.to(device)
            if neg_lengths.device != device:
                neg_lengths = neg_lengths.to(device)

        for p, c in edges:
            anchor = c
            pos = sims[anchor, p] / self.t

            # build negatives (prefer explicit candidates)
            if neg_pool is not None and neg_lengths is not None and anchor < neg_pool.size(0):
                pool = neg_pool[anchor]
                pool_len = int(neg_lengths[anchor].item())
                if pool_len <= 0:
                    continue
                num_neg_effective = self.num_neg if num_neg is None else num_neg
                if num_neg_effective is None or pool_len <= num_neg_effective:
                    neg_candidates = pool[:pool_len]
                else:
                    max_len = pool.size(0)
                    rand = torch.randint(0, max_len, (num_neg_effective,), device=device)
                    rand = rand % pool_len
                    neg_candidates = pool.gather(0, rand)
            elif neg_candidates_by_label is not None and anchor < len(neg_candidates_by_label):
                candidates = neg_candidates_by_label[anchor]
                neg_candidates = [j for j in candidates if j not in (anchor, p)]
                neg_candidates = torch.tensor(neg_candidates, device=device, dtype=torch.long) if neg_candidates else None
            elif same_level_map is not None and label_levels is not None and anchor < len(label_levels):
                lvl = label_levels[anchor]
                candidates = same_level_map.get(lvl, [])
                neg_candidates = [j for j in candidates if j not in (anchor, p)]
                neg_candidates = torch.tensor(neg_candidates, device=device, dtype=torch.long) if neg_candidates else None
            else:
                neg_candidates = None

            if neg_candidates is None or neg_candidates.numel() == 0:
                continue

            neg = sims[anchor, neg_candidates] / self.t  # [M]
            denom = torch.logsumexp(torch.cat([pos.view(1), neg], dim=0), dim=0)
            loss_terms.append(-(pos - denom))

        if not loss_terms:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(loss_terms).mean()


# -----------------------------
# Example helper to prepare tokenizer batch (for training)
# -----------------------------
def prepare_batch(
    encoder: SharedEncoder,
    texts: List[str],
    max_length: Optional[int] = None
) -> Dict[str, Tensor]:
    """
    Tokenize a batch of texts for training-time forward().
    """
    cfg = encoder.cfg
    toks = encoder.tokenizer(
        texts,
        max_length=max_length or cfg.max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(cfg.device) for k, v in toks.items()}


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    cfg = EncoderConfig()
    enc = SharedEncoder(cfg)
    sample_x = ["燕麥片", "牛肉歐姆蛋咖哩飯", "青醬義大利麵"]
    X = enc.encode_texts(sample_x, batch_size=2, progress=True)
    print("X:", X.shape)

    sample_labels = ["食品 > 全穀雜糧類 > 燕麥片", "食品 > 豆魚蛋肉類 > 牛肉", "食品 > 醬料類 > 青醬"]
    Z = enc.encode_labels(sample_labels, batch_size=3)
    print("Z:", Z.shape)
