# classifier.py
# RAE-HMC M3: Dual-Branch Hierarchical Classifier & Training Losses
# (Aligned with thesis §3.5)
# - Global branch (flat): logits_global -> sigmoid -> p_global
# - Local branch (level-wise): per-level heads -> concat logits_local -> p_local
# - Final classifier score (M3 output): p_cls = sigmoid(logits_global + logits_local)
# - Losses: Masked BCE / Focal, Alignment Loss, Path Hinge Loss
#
# NOTE:
#   * This module outputs classifier-side scores; fusion with memory scores (M2) is done in M4.
#   * For cl loss you must provide positive label indices per sample.
#   * For path hinge loss you must provide parent–child edge list.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class LossConfig:
    # Focal loss controls
    focal_alpha: Optional[float] = None
    focal_gamma: Optional[float] = None
    use_bce_loss: bool = True  # master switch: False -> skip BCE/Focal term

    # Alignment loss
    tau_cl: Optional[float] = None
    use_cl_loss: Optional[bool] = None
    use_inbatch_sample_neg_cl: bool = False

    num_neg_cl: Optional[int] = None

    # Path loss
    weight_path: Optional[float] = None
    use_path_loss: Optional[bool] = None
    path_on_local: Optional[bool] = None  # True -> use p_local; False -> use fused p_cls

    # Align/Path weights
    weight_cl: Optional[float] = None

    # Optional class weights
    pos_weight: Optional[Tensor] = None
    neg_weight: Optional[Tensor] = None

    def validate(self):
        required = {
            "tau_cl": self.tau_cl,
            "num_neg_cl": self.num_neg_cl,
            "use_cl_loss": self.use_cl_loss,
            "use_path_loss": self.use_path_loss,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "weight_cl": self.weight_cl,
            "weight_path": self.weight_path,
            "path_on_local": self.path_on_local,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"LossConfig missing required fields: {missing}. "
                "Provide them via TrainConfig/LossConfig to keep a single source of truth."
            )


@dataclass
class ClassifierConfig:
    hidden_size: int               # d, typically 768
    level_sizes: List[int]         # e.g., [L1, L2, L3, L4]
    dropout: Optional[float] = None           # optional dropout on h_x before heads
    local_num_heads: Optional[int] = None  # multi-head attention heads for conditional local branch
    # Global head config (fixed to MLP)
    global_hidden_ratio: Optional[float] = None  # hidden dim = ratio * hidden_size
    global_dropout: Optional[float] = None
    # Local head config
    local_head_mlp_from_level: Optional[int] = None  # 1-based; levels < this use linear, >= use MLP
    local_head_hidden_ratio: Optional[float] = None
    local_dropout: Optional[float] = None
    local_attn_mode: Optional[str] = None  # "prev_only" or "label_kv"
    # Fusion config (fixed to MLP)
    fusion_hidden_ratio: Optional[float] = None

    use_local_branch: Optional[bool] = None  # False -> global-only (flat) head
    use_global_branch: Optional[bool] = None  # False -> local-only head
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    level_slices: Optional[List[List[int]]] = None
    loss: Optional[LossConfig] = None

    def __post_init__(self):
        if self.dropout is None:
            raise ValueError("ClassifierConfig.dropout must be provided.")
        if self.global_hidden_ratio is None:
            raise ValueError("ClassifierConfig.global_hidden_ratio must be provided.")
        if self.local_head_mlp_from_level is None:
            raise ValueError("ClassifierConfig.local_head_mlp_from_level must be provided.")
        if self.local_head_hidden_ratio is None:
            raise ValueError("ClassifierConfig.local_head_hidden_ratio must be provided.")
        if self.local_attn_mode is None:
            raise ValueError("ClassifierConfig.local_attn_mode must be provided.")
        if self.fusion_hidden_ratio is None:
            raise ValueError("ClassifierConfig.fusion_hidden_ratio must be provided.")
        if self.loss is None:
            raise ValueError("ClassifierConfig.loss must be provided (LossConfig).")
        self.loss.validate()


# -----------------------------
# Dual-Branch Hierarchical Classifier
# -----------------------------
class DualBranchHierClassifier(nn.Module):
    """
    M3 classifier with Global (flat) + Local (level-wise) branches.
    Output:
        logits_global: [B, L]
        logits_local_concat: [B, L]
        logits_sum: [B, L]
        p_global, p_local, p_cls: [B, L] after sigmoid
    """
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.level_sizes = list(cfg.level_sizes)
        self.L = int(sum(self.level_sizes))
        self.use_global_branch = bool(getattr(cfg, "use_global_branch", True))
        self.use_local_branch = bool(getattr(cfg, "use_local_branch", True))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        # Global: linear or 1-hidden-layer MLP (flat, hierarchy-agnostic)
        global_hidden = max(1, int(cfg.hidden_size * cfg.global_hidden_ratio))
        global_drop = cfg.global_dropout if cfg.global_dropout is not None else cfg.dropout
        self.global_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, global_hidden),
            nn.GELU(),
            nn.Dropout(global_drop),
            nn.Linear(global_hidden, self.L),
        )

        # Local: hierarchical conditional branch (cross-attention per level)
        if self.use_local_branch:
            local_drop = cfg.local_dropout if cfg.local_dropout is not None else cfg.dropout
            local_heads = cfg.local_num_heads if cfg.local_num_heads is not None else 1
            d = cfg.hidden_size
            num_levels = len(self.level_sizes)
            self.local_attn_mode = str(getattr(cfg, "local_attn_mode", "prev_only")).lower().strip()

            # Level-1 representation: MLP(h)
            self.local_first = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(local_drop),
            )

            # Levels 2..L: Q = h (root), K/V = h_{l-1}
            self.local_attn = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d, num_heads=local_heads, dropout=local_drop, batch_first=True)
                for _ in range(max(0, num_levels))
            ])
            self.local_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(max(0, num_levels))])
            self.local_ff = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Dropout(local_drop),
                ) for _ in range(max(0, num_levels))
            ])

            # Per-level prediction heads (linear or MLP)
            mlp_from_level = int(getattr(cfg, "local_head_mlp_from_level", 1))
            local_head_hidden = max(1, int(cfg.hidden_size * cfg.local_head_hidden_ratio))
            self.local_heads = nn.ModuleList()
            for level_idx, n_l in enumerate(self.level_sizes, start=1):
                if level_idx >= mlp_from_level:
                    head = nn.Sequential(
                        nn.Linear(d, local_head_hidden),
                        nn.GELU(),
                        nn.Dropout(local_drop),
                        nn.Linear(local_head_hidden, n_l),
                    )
                else:
                    head = nn.Linear(d, n_l)
                self.local_heads.append(head)
        else:
            self.local_heads = nn.ModuleList()

        # Fusion: choose linear or shallow MLP on logits concat
        fusion_hidden_ratio = cfg.fusion_hidden_ratio if cfg.fusion_hidden_ratio is not None else 1.0
        fusion_drop = cfg.dropout if cfg.dropout is not None else 0.0
        fusion_hidden = max(1, int(2 * self.L * fusion_hidden_ratio))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.L, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_drop),
            nn.Linear(fusion_hidden, self.L),
        )
        self.level_slices = None
        if self.use_local_branch and getattr(cfg, "level_slices", None) is not None:
            # cached indices for index_copy
            self.level_slices = [torch.tensor(s, dtype=torch.long) for s in cfg.level_slices]

        self.to(cfg.device)

    def forward(self, h_x: Tensor, label_embeds: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Args:
            h_x: [B, d] shared encoder output (query embedding)
        Returns:
            dict of logits/probabilities for both branches and combined.
        """
        h = self.dropout(h_x)

        # Global (flat)
        logits_global = self.global_head(h) if self.use_global_branch else None  # [B, L]
        p_global = torch.sigmoid(logits_global) if logits_global is not None else None

        if self.use_local_branch:
            # Local (hierarchical conditional)
            h_levels: List[Tensor] = []

            # Level 1 base representation
            h_prev = self.local_first(h)                    # [B, d]

            if self.local_attn_mode == "label_kv" and label_embeds is not None and self.level_slices is not None:
                Z = label_embeds
                if Z.device != h.device:
                    Z = Z.to(h.device)
                for lvl_idx, lvl_ids in enumerate(self.level_slices):
                    if lvl_idx >= len(self.local_attn):
                        break
                    if lvl_ids.numel() == 0:
                        h_levels.append(h_prev)
                        continue
                    Z_l = Z.index_select(0, lvl_ids.to(h.device))  # [n_l, d]
                    q_base = h_prev
                    q_vec = h + q_base
                    q = q_vec.unsqueeze(1)                        # [B,1,d]
                    k = Z_l.unsqueeze(0).expand(h.size(0), -1, -1) # [B,n_l,d]
                    v = k
                    attn_out, _ = self.local_attn[lvl_idx](q, k, v)  # [B,1,d]
                    attn_out = attn_out.squeeze(1)                # [B,d]
                    attn_out = self.local_norm[lvl_idx](attn_out + q_base)
                    h_prev = self.local_ff[lvl_idx](attn_out)     # [B,d]
                    h_levels.append(h_prev)
            else:
                # Fallback: single-vector attention (prev-only)
                h_levels.append(h_prev)
                for idx in range(len(self.level_sizes) - 1):
                    if idx >= len(self.local_attn):
                        break
                    q = h.unsqueeze(1)                       # [B,1,d]
                    kv = h_levels[-1].unsqueeze(1)          # [B,1,d]
                    attn_out, _ = self.local_attn[idx](q, kv, kv)  # [B,1,d]
                    attn_out = attn_out.squeeze(1)          # [B,d]
                    attn_out = self.local_norm[idx](attn_out + h)  # residual on query
                    h_next = self.local_ff[idx](attn_out)   # [B,d]
                    h_levels.append(h_next)

            logits_local_list = [head(h_l) for head, h_l in zip(self.local_heads, h_levels)]  # per-level logits
            # Per-level sigmoid, then concat probabilities (per論文做法)
            p_local_list = [torch.sigmoid(lg) for lg in logits_local_list]

            if self.level_slices is not None:
                logits_local_concat = torch.zeros(h.size(0), self.L, device=h.device)
                p_local_concat = torch.zeros(h.size(0), self.L, device=h.device)
                for l_logits, p_lvl, idx in zip(logits_local_list, p_local_list, self.level_slices):
                    logits_local_concat.index_copy_(1, idx.to(h.device), l_logits)
                    p_local_concat.index_copy_(1, idx.to(h.device), p_lvl)
            else:
                logits_local_concat = torch.cat(logits_local_list, dim=-1)
                p_local_concat = torch.cat(p_local_list, dim=-1)
        else:
            logits_local_concat = torch.zeros_like(logits_global) if logits_global is not None else None
            p_local_concat = None
        p_local = p_local_concat

        # Fusion: choose logits- or prob-space fusion based on cfg
        if self.use_local_branch and self.use_global_branch:
            logits_concat = torch.cat([logits_local_concat, logits_global], dim=-1)  # [B, 2L], logits
            logits_fuse = self.fusion_mlp(logits_concat)
            logits_sum = logits_global + logits_local_concat + logits_fuse
            p_cls = torch.sigmoid(logits_sum)
        elif self.use_local_branch:
            logits_sum = logits_local_concat
            p_cls = p_local
        elif self.use_global_branch:
            logits_sum = logits_global
            p_cls = p_global
        else:
            raise ValueError("At least one of global/local branch must be enabled.")

        return {
            "logits_global": logits_global,
            "logits_local": logits_local_concat,
            "logits_sum": logits_sum,
            "p_global": p_global,
            "p_local": p_local,
            "p_cls": p_cls,
        }


# -----------------------------
# Losses (Aligned with §3.5)
def focal_loss(
    logits: Tensor,             # [B, L] logits before sigmoid
    Y: Tensor,                  # [B, L] multi-hot ground truth (1/0)
    mask: Optional[Tensor] = None,   # [B, L]
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """
    Multi-label focal loss in logit space (stable BCE-with-logits formulation).
    """
    if mask is None:
        mask = torch.ones_like(Y)

    bce = F.binary_cross_entropy_with_logits(logits, Y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * Y + (1.0 - p) * (1.0 - Y)
    alpha_t = alpha * Y + (1.0 - alpha) * (1.0 - Y)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def cl_loss(
    h_x: Tensor,
    Z: Tensor,
    pos_indices_per_sample: List[List[int]],
    pos_weights_per_sample: Optional[List[List[float]]] = None,
    candidate_neg_indices: Optional[object] = None,
    temperature: float = 0.07,
    same_level_map: Optional[Dict[int, List[int]]] = None,
    label_levels: Optional[List[int]] = None,
    level_sizes: Optional[List[int]] = None,
    max_negatives: Optional[int] = 64,
    y_true: Optional[Tensor] = None,
    use_inbatch_sample_neg: bool = False,
) -> Tensor:
    """InfoNCE-style cl loss with per-level negative pools."""
    device = h_x.device
    B, _ = h_x.shape
    L = Z.shape[0]

    q = F.normalize(h_x, p=2, dim=-1)
    Z_norm = F.normalize(Z, p=2, dim=-1)

    loss_terms: List[Tensor] = []
    inbatch_neg_mask: Optional[Tensor] = None
    if use_inbatch_sample_neg and y_true is not None and label_levels is not None:
        lvl_t = torch.tensor(label_levels, device=device, dtype=torch.long)
        eff = torch.ones_like(lvl_t, dtype=torch.bool, device=device)
        if level_sizes is not None:
            lvl_sizes_t = torch.tensor(level_sizes, device=device, dtype=torch.long)
            num_levels = lvl_sizes_t.numel()
            lvl_idx = lvl_t
            if num_levels > 0 and lvl_t.numel() > 0:
                if lvl_t.max().item() == num_levels and lvl_t.min().item() >= 1:
                    lvl_idx = lvl_t - 1
            valid = (lvl_idx >= 0) & (lvl_idx < num_levels)
            size_mask = torch.zeros_like(valid, dtype=torch.bool)
            if valid.any():
                size_ok = (lvl_sizes_t.index_select(0, lvl_idx[valid]) > 1)
                size_mask[valid] = size_ok
            eff = eff & size_mask
        if eff.any():
            yb = (y_true > 0.5).to(torch.float32)
            y_eff = yb[:, eff]
            overlap = (y_eff @ y_eff.T) > 0
            valid = (y_eff.sum(dim=1) > 0)
            inbatch_neg_mask = (~overlap) & valid.unsqueeze(0) & valid.unsqueeze(1)
            inbatch_neg_mask.fill_diagonal_(False)

    for i in range(B):
        pos_idx = pos_indices_per_sample[i]
        if not pos_idx:
            continue
        weights = None
        weight_by_label = None
        if pos_weights_per_sample is not None and i < len(pos_weights_per_sample):
            w_list = pos_weights_per_sample[i]
            if w_list and len(w_list) == len(pos_idx):
                weights = torch.tensor(w_list, device=device, dtype=q.dtype)
                weight_by_label = {lab: float(w_list[idx]) for idx, lab in enumerate(pos_idx)}
        sim_neg_parts: List[Tensor] = []

        if inbatch_neg_mask is not None:
            neg_sample_idx = torch.nonzero(inbatch_neg_mask[i], as_tuple=False).squeeze(-1)
            if neg_sample_idx.numel() > 0:
                sim_neg_sample = (q[i:i+1] @ q.index_select(0, neg_sample_idx).T).squeeze(0)
                sim_neg_parts.append(sim_neg_sample)

        if candidate_neg_indices is not None:
            if torch.is_tensor(candidate_neg_indices):
                neg_idx = candidate_neg_indices[i]
                neg_idx = neg_idx[neg_idx >= 0]
            else:
                neg_idx = [j for j in candidate_neg_indices[i] if j not in pos_idx]
                neg_idx = torch.tensor(neg_idx, device=device, dtype=torch.long) if neg_idx else None
            if neg_idx is not None and neg_idx.numel() > 0:
                sim_neg = (q[i:i+1] @ Z_norm[neg_idx].T).squeeze(0)
                sim_neg_parts.append(sim_neg)
            if not sim_neg_parts:
                continue
            sim_pos = (q[i:i+1] @ Z_norm[pos_idx].T).squeeze(0)
            sim_neg = torch.cat(sim_neg_parts, dim=0)
            for sp_idx, sp in enumerate(sim_pos):
                w = weights[sp_idx] if weights is not None else 1.0
                num = sp / temperature
                den = torch.logsumexp(torch.cat([sp.view(1), sim_neg]) / temperature, dim=0)
                loss_terms.append(-(num - den) * w)
            continue

        if same_level_map is None or label_levels is None:
            continue

        level_to_pos: Dict[int, List[int]] = {}
        for p in pos_idx:
            if 0 <= p < len(label_levels):
                lvl = label_levels[p]
                level_to_pos.setdefault(lvl, []).append(p)

        for lvl, pos_lvl in level_to_pos.items():
            excl = set(pos_lvl)
            base_candidates = [j for j in same_level_map.get(lvl, []) if j not in excl]
            if not base_candidates:
                continue
            if max_negatives is not None and len(base_candidates) > max_negatives:
                base_candidates = random.sample(base_candidates, max_negatives)

            sim_pos = (q[i:i+1] @ Z_norm[pos_lvl].T).squeeze(0)
            sim_neg_parts_lvl = list(sim_neg_parts)
            sim_neg_parts_lvl.append((q[i:i+1] @ Z_norm[base_candidates].T).squeeze(0))
            sim_neg = torch.cat(sim_neg_parts_lvl, dim=0)
            for sp_idx, sp in enumerate(sim_pos):
                if weight_by_label is not None:
                    w = weight_by_label.get(pos_lvl[sp_idx], 1.0)
                else:
                    w = 1.0
                num = sp / temperature
                den = torch.logsumexp(torch.cat([sp.view(1), sim_neg]) / temperature, dim=0)
                loss_terms.append(-(num - den) * w)

    if not loss_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(loss_terms).mean()


def path_hinge_loss(
    p_probs: Tensor,                 # [B, L] probabilities per thesis
    edges_parent_child: List[Tuple[int, int]],
) -> Tensor:
    """
    Path hinge loss aligned to paper Eq.(9):
        sum_{(p,c)} max(0, s_c - s_p)
    Uses probabilities directly (no logits/extra margin).
    """
    s = p_probs  # already probabilities
    terms = []
    for (p, c) in edges_parent_child:
        terms.append((s[:, c] - s[:, p]).clamp(min=0.0))
    if not terms:
        return torch.tensor(0.0, device=p_probs.device, requires_grad=True)
    return torch.stack(terms, dim=0).mean()


# -----------------------------
# Trainer-friendly wrapper
# -----------------------------
class JointLossCombiner(nn.Module):
    """
    Combine BCE/Focal + Align + Path into total loss (禮3.5).
    Align uses (h_x, Z) with positive label indices per sample.
    """
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.loss = cfg.loss

    def forward(
        self,
        # classifier outputs
        p_cls: Tensor,                        # [B, L]
        # supervision
        Y: Tensor,                            # [B, L]
        mask: Optional[Tensor],               # [B, L] or None
        # cl resources
        h_x: Tensor,                          # [B, d]
        Z: Tensor,                            # [L, d]
        pos_indices_per_sample: List[List[int]],
        # path resources
        edges_parent_child: List[Tuple[int, int]],
        pos_weights_per_sample: Optional[List[List[float]]] = None,
        # misc
        logits_cls: Optional[Tensor] = None,  # [B, L], optional for logit-space focal
        same_level_map: Optional[Dict[int, List[int]]] = None,
        label_levels: Optional[List[int]] = None,
        candidate_neg_indices: Optional[List[List[int]]] = None,
        p_local: Optional[Tensor] = None,     # [B, L] local branch probs (optional; used for path loss)
    ) -> Dict[str, Tensor]:
        if getattr(self.loss, "use_bce_loss", True):
            if logits_cls is None:
                eps = 1e-8
                p_safe = p_cls.clamp(min=eps, max=1.0 - eps)
                logits_cls = torch.log(p_safe / (1.0 - p_safe))
            loss_bce = focal_loss(
                logits_cls, Y, mask=mask,
                alpha=self.loss.focal_alpha,
                gamma=self.loss.focal_gamma,
            )
        else:
            loss_bce = torch.tensor(0.0, device=p_cls.device, requires_grad=True)

        loss_cl = torch.tensor(0.0, device=p_cls.device, requires_grad=True)
        if getattr(self.loss, "use_cl_loss", True) and getattr(self.loss, "weight_cl", 0.0) != 0:
            loss_cl = cl_loss(
                h_x, Z, pos_indices_per_sample,
                pos_weights_per_sample=pos_weights_per_sample,
                candidate_neg_indices=candidate_neg_indices,
                temperature=self.loss.tau_cl,
                same_level_map=same_level_map,
                label_levels=label_levels,
                max_negatives=self.loss.num_neg_cl if getattr(self.loss, "num_neg_cl", None) is not None else None,
                y_true=Y,
                use_inbatch_sample_neg=getattr(self.loss, "use_inbatch_sample_neg_cl", False),
                level_sizes=self.cfg.level_sizes,
            )

        if getattr(self.loss, "use_path_loss", True) and getattr(self.loss, "weight_path", 0.0) != 0:
            loss_path = path_hinge_loss(p_cls, edges_parent_child)
        else:
            loss_path = torch.tensor(0.0, device=p_cls.device)

        total = loss_bce             + self.loss.weight_cl * loss_cl             + self.loss.weight_path * loss_path

        return {
            "loss_total": total,
            "loss_bce": loss_bce.detach(),
            "loss_cl": loss_cl.detach(),
            "loss_path": loss_path.detach(),
        }


# -----------------------------
# Minimal smoke test (optional)

# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shapes
    B, d = 4, 16
    level_sizes = [3, 4]   # L = 7
    L = sum(level_sizes)

    # Fake encoder outputs and label embeddings
    h_x = F.normalize(torch.randn(B, d, device=device), p=2, dim=-1)
    Z = F.normalize(torch.randn(L, d, device=device), p=2, dim=-1)

    # Ground truth multi-hot Y and mask
    Y = torch.zeros(B, L, device=device)
    Y[0, [0, 4]] = 1
    Y[1, [1]] = 1
    Y[2, [2, 5]] = 1
    Y[3, [6]] = 1
    mask = torch.ones(B, L, device=device)

    # Parent-child edges (indices in concatenated label order)
    edges_pc = [(0, 3), (1, 4), (2, 5)]  # parents in L1, children in L2 (toy)

    # Positives per sample for cl
    pos_idx = [[0, 4], [1], [2, 5], [6]]

    cfg = ClassifierConfig(
        hidden_size=d,
        level_sizes=level_sizes,
        dropout=0.0,
        global_hidden_ratio=1.0,
        local_head_mlp_from_level=2,
        local_head_hidden_ratio=0.5,
        local_attn_mode="label_kv",
        fusion_hidden_ratio=1.0,
        global_dropout=None,
        local_dropout=None,
        use_local_branch=True,
        device=device,
        loss=LossConfig(
            focal_alpha=0.7,
            focal_gamma=0.0,
            use_bce_loss=True,
            path_on_local=True,
            tau_cl=0.07,
            num_neg_cl=16,
            use_cl_loss=True,
            use_path_loss=True,
            weight_cl=1.0,
            weight_path=1.0,
        ),
    )

    clf = DualBranchHierClassifier(cfg).to(device)
    out = clf(h_x)
    print("logits_sum:", tuple(out["logits_sum"].shape), "p_cls:", tuple(out["p_cls"].shape))

    # Combine losses
    comb = JointLossCombiner(cfg).to(device)
    losses = comb(
        p_cls=out["p_cls"],
        logits_cls=out["logits_sum"],
        Y=Y,
        mask=mask,
        h_x=h_x,
        Z=Z,
        pos_indices_per_sample=pos_idx,
        candidate_neg_indices=None,
        edges_parent_child=edges_pc,
        same_level_map=None,
        label_levels=None,
    )
    print({k: float(v) for k, v in losses.items()})
