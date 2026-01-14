# classifier.py
# RAE-HMC M3: Dual-Branch Hierarchical Classifier & Training Losses
# (Aligned with thesis §3.5)
# - Global branch (flat): logits_global -> sigmoid -> p_global
# - Local branch (level-wise): per-level heads -> concat logits_local -> p_local
# - Final classifier score (M3 output): p_cls = sigmoid(logits_global + logits_local)
# - Losses: Masked BCE / Focal, Alignment Loss, Path Hinge Loss, Label Loss (HCL), Sample Contrast
#
# NOTE:
#   * This module outputs classifier-side scores; fusion with memory scores (M2) is done in M4.
#   * For alignment loss you must provide positive label indices per sample.
#   * For path hinge loss you must provide parent–child edge list.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the HCL from M1 (encoder.py) for joint training (per §3.5)
try:
    from encoder import HierarchicalContrastiveLoss
except Exception:
    HierarchicalContrastiveLoss = None  # Optional: user may choose to import later


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
    tau_align: Optional[float] = None
    use_align_loss: Optional[bool] = None

    # Label contrastive loss (formerly HCL)
    tau_label: Optional[float] = None
    num_neg: Optional[int] = None
    use_label_loss: Optional[bool] = None

    # Path loss
    weight_path: Optional[float] = None
    use_path_loss: Optional[bool] = None
    path_on_local: Optional[bool] = None  # True -> use p_local; False -> use fused p_cls

    # Align/Path/Label weights
    weight_label: Optional[float] = None
    weight_align: Optional[float] = None

    # Sample-sample contrastive
    weight_sample_contrast: Optional[float] = None
    tau_sample_contrast: Optional[float] = None
    use_sample_loss: Optional[bool] = None
    sample_repeat: Optional[int] = None
    sample_queue_size: Optional[int] = None
    exclude_same_level_overlap_neg: Optional[bool] = None

    # Optional class weights
    pos_weight: Optional[Tensor] = None
    neg_weight: Optional[Tensor] = None

    def validate(self):
        required = {
            "tau_align": self.tau_align,
            "tau_label": self.tau_label,
            "num_neg": self.num_neg,
            "use_align_loss": self.use_align_loss,
            "use_label_loss": self.use_label_loss,
            "use_path_loss": self.use_path_loss,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "weight_label": self.weight_label,
            "weight_align": self.weight_align,
            "weight_path": self.weight_path,
            "weight_sample_contrast": self.weight_sample_contrast,
            "tau_sample_contrast": self.tau_sample_contrast,
            "use_sample_loss": self.use_sample_loss,
            "sample_repeat": self.sample_repeat,
            "sample_queue_size": self.sample_queue_size,
            "exclude_same_level_overlap_neg": self.exclude_same_level_overlap_neg,
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
    dropout: float = 0.1           # optional dropout on h_x before heads
    local_num_heads: int = 2       # multi-head attention heads for conditional local branch
    # Global head config
    global_hidden_ratio: float = 0.5   # hidden dim = ratio * hidden_size
    global_dropout: Optional[float] = None

    use_local_branch: Optional[bool] = None  # False -> global-only (flat) head
    use_global_branch: Optional[bool] = None  # False -> local-only head
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    level_slices: Optional[List[List[int]]] = None
    loss: Optional[LossConfig] = None

    def __post_init__(self):
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

        # Global: single linear head (flat, hierarchy-agnostic)
        self.global_head = nn.Linear(cfg.hidden_size, self.L)

        # Local: hierarchical conditional branch (cross-attention per level)
        if self.use_local_branch:
            d = cfg.hidden_size
            num_levels = len(self.level_sizes)

            # Level-1 representation: MLP(h)
            self.local_first = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

            # Levels 2..L: Q = h (root), K/V = h_{l-1}
            self.local_attn = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d, num_heads=cfg.local_num_heads, dropout=cfg.dropout, batch_first=True)
                for _ in range(max(0, num_levels - 1))
            ])
            self.local_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(max(0, num_levels - 1))])
            self.local_ff = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ) for _ in range(max(0, num_levels - 1))
            ])

            # Per-level prediction heads
            self.local_heads = nn.ModuleList([
                nn.Linear(d, n_l) for n_l in self.level_sizes
            ])
        else:
            self.local_heads = nn.ModuleList()

        # Fusion MLP: concat -> MLP -> fused logits
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.L, self.L),
        )
        # Logit-space fusion (residual-style): logits_global + logits_local + linear([logits_local; logits_global])
        self.fusion_linear_logits = nn.Linear(2 * self.L, self.L)

        self.level_slices = None
        if self.use_local_branch and getattr(cfg, "level_slices", None) is not None:
            # cached indices for index_copy
            self.level_slices = [torch.tensor(s, dtype=torch.long) for s in cfg.level_slices]

        self.to(cfg.device)

    def forward(self, h_x: Tensor) -> Dict[str, Tensor]:
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

            # Level 1 representation
            h1 = self.local_first(h)                    # [B, d]
            h_levels.append(h1)

            # Levels 2..: cross-attention with root query
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
            # Original logit-space fusion (allows cross-label mixing via fusion_linear_logits):
            # logits_sum = logits_global + logits_local + Linear([logits_local; logits_global])
            logits_concat = torch.cat([logits_local_concat, logits_global], dim=-1)  # [B, 2L], logits
            logits_fuse = self.fusion_linear_logits(logits_concat)
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
    p: Tensor,             # [B, L] probabilities after sigmoid
    Y: Tensor,             # [B, L] multi-hot ground truth (1/0)
    mask: Optional[Tensor] = None,   # [B, L]
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """
    Multi-label focal loss with optional mask (per Lin et al.).
    """
    eps = 1e-8
    if mask is None:
        mask = torch.ones_like(p)

    pt_pos = p.clamp(min=eps, max=1.0 - eps)
    pt_neg = (1.0 - p).clamp(min=eps, max=1.0 - eps)

    loss_pos = -alpha * (1 - pt_pos).pow(gamma) * Y * torch.log(pt_pos)
    loss_neg = -(1 - alpha) * (pt_neg).pow(gamma) * (1 - Y) * torch.log(pt_neg)

    loss = (loss_pos + loss_neg) * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def alignment_loss(
    h_x: Tensor,
    Z: Tensor,
    pos_indices_per_sample: List[List[int]],
    candidate_neg_indices: Optional[List[List[int]]] = None,
    temperature: float = 0.07,
    same_level_map: Optional[Dict[int, List[int]]] = None,
    label_levels: Optional[List[int]] = None,
    forbid_relatives: Optional[Dict[int, Set[int]]] = None,
    max_negatives: Optional[int] = 64,
) -> Tensor:
    """InfoNCE-style alignment loss with hierarchy-aware negative sampling."""
    device = h_x.device
    B, _ = h_x.shape
    L = Z.shape[0]

    q = F.normalize(h_x, p=2, dim=-1)
    Z_norm = F.normalize(Z, p=2, dim=-1)

    loss_terms: List[Tensor] = []
    for i in range(B):
        pos_idx = pos_indices_per_sample[i]
        if not pos_idx:
            continue

        relatives: Set[int] = set()
        if forbid_relatives is not None:
            for p in pos_idx:
                relatives.update(forbid_relatives.get(p, set()))

        if candidate_neg_indices is None:
            excl = set(pos_idx)
            base_candidates: List[int] = []

            if same_level_map is not None and label_levels is not None:
                levels = {label_levels[p] for p in pos_idx if 0 <= p < len(label_levels)}
                for lvl in levels:
                    lvl_candidates = same_level_map.get(lvl, [])
                    base_candidates.extend([j for j in lvl_candidates if j not in excl and j not in relatives])

            if not base_candidates:
                base_candidates = [j for j in range(L) if j not in excl and j not in relatives]

            seen: Set[int] = set()
            base_candidates = [j for j in base_candidates if not (j in seen or seen.add(j))]

            if max_negatives is not None:
                if len(base_candidates) > max_negatives:
                    base_candidates = random.sample(base_candidates, max_negatives)
                elif len(base_candidates) < max_negatives:
                    extra_pool = [j for j in range(L) if j not in excl and j not in relatives and j not in base_candidates]
                    if extra_pool:
                        extra_take = min(max_negatives - len(base_candidates), len(extra_pool))
                        base_candidates.extend(random.sample(extra_pool, extra_take))
            neg_idx = base_candidates
        else:
            neg_idx = [j for j in candidate_neg_indices[i] if j not in pos_idx and j not in relatives]

        if len(neg_idx) == 0:
            continue

        sim_pos = (q[i:i+1] @ Z_norm[pos_idx].T).squeeze(0)
        sim_neg = (q[i:i+1] @ Z_norm[neg_idx].T).squeeze(0)

        for sp in sim_pos:
            num = sp / temperature
            den = torch.logsumexp(torch.cat([sp.view(1), sim_neg]) / temperature, dim=0)
            loss_terms.append(-(num - den))

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


def sample_contrastive_loss(
    h_x: Tensor,          # [B, d]
    Y: Tensor,            # [B, L]
    label_levels: Optional[List[int]] = None,
    temperature: float = 0.07,
    repeat_times: int = 1,  # sampling rounds per batch
    queue_feats: Optional[Tensor] = None,     # [Q, d] optional FIFO queue embeddings (normalized)
    queue_labels: Optional[Tensor] = None,    # [Q, L] labels for queue entries
    exclude_anchor_overlap: bool = False,     # if True, drop negatives sharing anchor's other same-level labels
    extra_pool_feats: Optional[Tensor] = None,   # [E, d] optional extra samples (e.g., from inverted index)
    extra_pool_labels: Optional[Tensor] = None,  # [E, L] labels for extra samples
    extra_pos_candidates_map: Optional[List[Dict[int, List[int]]]] = None,  # per-anchor label->pos indices
) -> Tensor:
    """
    Sample-sample contrastive loss (HMCL-like, BCE on sigmoid similarities).
    Level-wise + label-wise sampling:
      * For each anchor sample and each active label j (level l):
          - Positives: 1 sample (uniform) that also has label j (same layer).
          - Negatives: 1 sample that (i) does NOT have label j,
                       (ii) has at least one label in the same level l.
      * Optionally repeat the sampling repeat_times to enlarge contrastive pairs (paper appendix).
      * Loss per label j: -mean(logsigmoid(sim_pos/?)) - mean(logsigmoid(-sim_neg/?))
    """
    device = h_x.device
    B, L = Y.shape
    if label_levels is None or len(label_levels) != L:
        raise ValueError(
            "sample_contrastive_loss requires label_levels with length = number of labels (L); "
            f"got {None if label_levels is None else len(label_levels)} vs {L}."
        )

    h_norm = F.normalize(h_x, p=2, dim=-1)

    use_direct_pos = extra_pos_candidates_map is not None
    # Positive pool: only extra positives (e.g., inverted index); do not include current batch
    if extra_pool_feats is not None and extra_pool_feats.numel() > 0:
        pos_pool_feats = F.normalize(extra_pool_feats, p=2, dim=-1)
    else:
        pos_pool_feats = h_norm.new_zeros((0, h_norm.size(1)))
    if not use_direct_pos:
        if extra_pool_labels is not None and extra_pool_labels.numel() > 0:
            pos_pool_labels = extra_pool_labels
        else:
            pos_pool_labels = Y.new_zeros((0, Y.size(1)))
    else:
        if pos_pool_feats.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

    # Negative pool: current batch + optional queue (detach queue to avoid grads)
    neg_pool_feats = h_norm
    neg_pool_labels = Y
    if queue_feats is not None and queue_labels is not None and queue_feats.numel() > 0:
        q_feats = queue_feats.detach()
        q_labels = queue_labels.detach()
        neg_pool_feats = torch.cat([neg_pool_feats, q_feats], dim=0)
        neg_pool_labels = torch.cat([neg_pool_labels, q_labels], dim=0)

    level_to_labels: Dict[int, List[int]] = {}
    for idx, lvl in enumerate(label_levels):
        level_to_labels.setdefault(lvl, []).append(idx)

    idx_neg_all = torch.arange(neg_pool_feats.size(0), device=device)

    all_pos_terms: List[Tensor] = []
    all_neg_terms: List[Tensor] = []

    for _ in range(max(1, repeat_times)):
        for i in range(B):
            active_labels = (Y[i] > 0.5).nonzero(as_tuple=False).flatten().tolist()
            if not active_labels:
                continue
            active_label_set = set(active_labels)
            for j in active_labels:
                lvl = label_levels[j]
                lvl_labels = level_to_labels.get(lvl, [])

                # Positive: one sample sharing label j from inverted-index pool
                if use_direct_pos:
                    pos_map = extra_pos_candidates_map[i] if (extra_pos_candidates_map is not None and i < len(extra_pos_candidates_map)) else None
                    if not pos_map:
                        continue
                    pos_indices = pos_map.get(j, None)
                    if not pos_indices:
                        continue
                    pick = int(torch.randint(0, len(pos_indices), (1,), device=device).item())
                    pos_idx = pos_indices[pick]
                    pos_vec = pos_pool_feats[pos_idx:pos_idx+1]
                else:
                    pos_mask = (pos_pool_labels[:, j] > 0.5)
                    if pos_mask.sum() == 0:
                        continue
                    pos_idx_all = pos_mask.nonzero(as_tuple=False).squeeze(1)
                    pos_idx = pos_idx_all[torch.randint(0, pos_idx_all.numel(), (1,))]
                    pos_vec = pos_pool_feats[pos_idx]

                # Negative: choose a negative label from the same level, then sample a negative instance.
                overlap_mask = None
                if exclude_anchor_overlap:
                    anchor_same_level = (Y[i, lvl_labels] > 0.5)
                    overlap_mask = ((neg_pool_labels[:, lvl_labels] > 0.5) & anchor_same_level.unsqueeze(0)).any(dim=1)

                neg_label_indices: List[Tensor] = []
                for k in lvl_labels:
                    if k == j or k in active_label_set:
                        continue
                    cand_mask = (neg_pool_labels[:, k] > 0.5) & (neg_pool_labels[:, j] < 0.5) & (idx_neg_all != i)
                    if overlap_mask is not None:
                        cand_mask = cand_mask & (~overlap_mask)
                    if cand_mask.sum() == 0:
                        continue
                    neg_label_indices.append(cand_mask.nonzero(as_tuple=False).squeeze(1))

                if not neg_label_indices:
                    continue

                q = h_norm[i:i+1]                  # [1, d]
                sim_pos = (q @ pos_vec.t()).squeeze(0) / temperature  # scalar
                all_pos_terms.append(sim_pos)

                # draw one negative (1:1) from same-level negative labels
                num_neg_labels = len(neg_label_indices)
                label_pick = int(torch.randint(0, num_neg_labels, (1,), device=device).item())
                cand_indices = neg_label_indices[label_pick]
                idx_pick = cand_indices[torch.randint(0, cand_indices.numel(), (1,), device=device)]
                neg_vec = neg_pool_feats[idx_pick:idx_pick+1]  # [1, d]
                sim_neg = (q @ neg_vec.t()).squeeze(0) / temperature
                all_neg_terms.append(sim_neg)

    if not all_pos_terms and not all_neg_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    if all_pos_terms:
        pos_cat = torch.cat(all_pos_terms)
        loss = loss - torch.mean(F.logsigmoid(pos_cat))
    if all_neg_terms:
        neg_cat = torch.cat(all_neg_terms)
        loss = loss - torch.mean(F.logsigmoid(-neg_cat))
    return loss


# -----------------------------
# Trainer-friendly wrapper
# -----------------------------
class JointLossCombiner(nn.Module):
    """
    Combine BCE/Focal + λ1*Label + λ2*Align + λ3*Path + λ4*Sample into total loss (§3.5).
    Label loss (hierarchical contrast) uses label embeddings Z and parent–child edges.
    Align uses (h_x, Z) with positive label indices per sample.
    """
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.loss = cfg.loss
        if HierarchicalContrastiveLoss is not None:
            self.label_loss_fn = HierarchicalContrastiveLoss(temperature=self.loss.tau_label, num_neg=32)
        else:
            self.label_loss_fn = None  # user can set later
        # FIFO queue for sample contrast (optional)
        self.sample_queue_feats: Optional[Tensor] = None
        self.sample_queue_labels: Optional[Tensor] = None

    def forward(
        self,
        # classifier outputs
        p_cls: Tensor,                        # [B, L]
        # supervision
        Y: Tensor,                            # [B, L]
        mask: Optional[Tensor],               # [B, L] or None
        # alignment resources
        h_x: Tensor,                          # [B, d]
        Z: Tensor,                            # [L, d]
        pos_indices_per_sample: List[List[int]],
        # path resources
        edges_parent_child: List[Tuple[int, int]],
        # Label contrast resources (optional)
        Z_for_label_loss: Optional[Tensor] = None,   # [L, d], if None -> use Z
        same_level_map: Optional[Dict[int, List[int]]] = None,
        label_levels: Optional[List[int]] = None,
        forbid_relatives: Optional[Dict[int, Set[int]]] = None,
        p_local: Optional[Tensor] = None,     # [B, L] local branch probs (optional; used for path loss)
        # Optional extra positives for sample contrast (e.g., from inverted index)
        extra_pos_feats: Optional[Tensor] = None,   # [E, d]
        extra_pos_labels: Optional[Tensor] = None,  # [E, L]
        extra_pos_candidates_map: Optional[List[Dict[int, List[int]]]] = None,  # per-anchor label->pos indices
    ) -> Dict[str, Tensor]:
        # Primary classification loss: always focal (can be disabled via use_bce_loss=False)
        if getattr(self.loss, "use_bce_loss", True):
            loss_bce = focal_loss(
                p_cls, Y, mask=mask,
                alpha=self.loss.focal_alpha,
                gamma=self.loss.focal_gamma,
            )
        else:
            loss_bce = torch.tensor(0.0, device=p_cls.device, requires_grad=True)

        # Align
        loss_align = torch.tensor(0.0, device=p_cls.device, requires_grad=True)
        if getattr(self.loss, "use_align_loss", True) and getattr(self.loss, "weight_align", 0.0) != 0:
            loss_align = alignment_loss(
                h_x, Z, pos_indices_per_sample,
                candidate_neg_indices=None,
                temperature=self.loss.tau_align,
                same_level_map=same_level_map,
                label_levels=label_levels,
                forbid_relatives=forbid_relatives,
                max_negatives=self.loss.num_neg if getattr(self.loss, "num_neg", None) is not None else None,
            )

        # Path hinge
        if getattr(self.loss, "use_path_loss", True) and getattr(self.loss, "weight_path", 0.0) != 0:
            # Simplified: always apply path constraint on final classifier probabilities p_cls.
            loss_path = path_hinge_loss(p_cls, edges_parent_child)
        else:
            loss_path = torch.tensor(0.0, device=p_cls.device)

        # Label contrast (optional, can be zero if not provided)
        loss_label = torch.tensor(0.0, device=p_cls.device)
        if self.label_loss_fn is not None and Z is not None and edges_parent_child and getattr(self.loss, "use_label_loss", True) and getattr(self.loss, "weight_label", 0.0) != 0:
            label_num_neg = getattr(self.loss, "num_neg", None)
            loss_label = self.label_loss_fn(
                Z_for_label_loss if Z_for_label_loss is not None else Z,
                edges_parent_child,
                same_level_map=same_level_map,
                label_levels=label_levels,
                forbid_relatives=forbid_relatives,
                num_neg=label_num_neg,
            )

        # Sample-sample contrast (optional, can be disabled via use_sample_loss)
        loss_sample = torch.tensor(0.0, device=p_cls.device, requires_grad=True)
        if getattr(self.loss, "use_sample_loss", True) and getattr(self.loss, "weight_sample_contrast", 0.0) != 0:
            h_x_for_contrast = F.normalize(h_x, p=2, dim=-1)
            extra_pool_feats = extra_pos_feats
            if extra_pool_feats is not None and extra_pool_feats.numel() > 0:
                # Keep positive/negative pools in the same space.
                extra_pool_feats = F.normalize(extra_pool_feats, p=2, dim=-1)
            loss_sample = sample_contrastive_loss(
                h_x_for_contrast, Y,
                label_levels=label_levels,
                temperature=self.loss.tau_sample_contrast,
                repeat_times=max(1, int(getattr(self.loss, "sample_repeat", 1))),
                queue_feats=self.sample_queue_feats,
                queue_labels=self.sample_queue_labels,
                exclude_anchor_overlap=getattr(self.loss, "exclude_same_level_overlap_neg", False),
                extra_pool_feats=extra_pool_feats,
                extra_pool_labels=extra_pos_labels,
                extra_pos_candidates_map=extra_pos_candidates_map,
            )
            if not torch.isfinite(loss_sample):
                loss_sample = torch.tensor(0.0, device=p_cls.device)

            # Update FIFO queue for sample contrast
            if getattr(self.loss, "sample_queue_size", 0) > 0:
                with torch.no_grad():
                    feats_new = h_x_for_contrast.detach()
                    labels_new = Y.detach()
                    if self.sample_queue_feats is None or self.sample_queue_labels is None:
                        self.sample_queue_feats = feats_new
                        self.sample_queue_labels = labels_new
                    else:
                        self.sample_queue_feats = torch.cat([self.sample_queue_feats, feats_new], dim=0)
                        self.sample_queue_labels = torch.cat([self.sample_queue_labels, labels_new], dim=0)
                    max_q = self.loss.sample_queue_size
                    if self.sample_queue_feats.size(0) > max_q:
                        self.sample_queue_feats = self.sample_queue_feats[-max_q:]
                        self.sample_queue_labels = self.sample_queue_labels[-max_q:]

        total = loss_bce \
            + self.loss.weight_align * loss_align \
            + self.loss.weight_path * loss_path \
            + self.loss.weight_label * loss_label \
            + self.loss.weight_sample_contrast * loss_sample

        return {
            "loss_total": total,
            "loss_bce": loss_bce.detach(),
            "loss_align": loss_align.detach(),
            "loss_path": loss_path.detach(),
            "loss_label": loss_label.detach(),
            "loss_sample": loss_sample.detach(),
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

    # Positives per sample for alignment
    pos_idx = [[0, 4], [1], [2, 5], [6]]

    cfg = ClassifierConfig(
        hidden_size=d,
        level_sizes=level_sizes,
        dropout=0.0,
        use_local_branch=True,
        device=device,
        loss=LossConfig(
            focal_alpha=0.7,
            focal_gamma=0.0,
            use_bce_loss=True,
            path_on_local=True,
            tau_align=0.07,
            tau_label=0.07,
            num_neg=16,
            use_align_loss=True,
            use_label_loss=True,
            use_path_loss=True,
            weight_label=1.0,
            weight_align=1.0,
            weight_path=1.0,
            weight_sample_contrast=0.1,
            tau_sample_contrast=0.07,
            use_sample_loss=True,
            sample_repeat=1,
            sample_queue_size=0,
            exclude_same_level_overlap_neg=False,
        ),
    )

    clf = DualBranchHierClassifier(cfg).to(device)
    out = clf(h_x)
    print("logits_sum:", tuple(out["logits_sum"].shape), "p_cls:", tuple(out["p_cls"].shape))

    # Combine losses
    comb = JointLossCombiner(cfg).to(device)
    losses = comb(
        p_cls=out["p_cls"],
        Y=Y,
        mask=mask,
        h_x=h_x,
        Z=Z,
        pos_indices_per_sample=pos_idx,
        edges_parent_child=edges_pc,
        Z_for_label_loss=Z,  # (optional) use same Z
        same_level_map=None,
        label_levels=None,
        forbid_relatives=None,
    )
    print({k: float(v) for k, v in losses.items()})
