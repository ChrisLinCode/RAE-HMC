# classifier.py
# RAE-HMC M3: Dual-Branch Hierarchical Classifier & Training Losses
# (Aligned with thesis §3.5)
# - Global branch (flat): logits_global -> sigmoid -> p_global
# - Local branch (level-wise): per-level heads -> concat logits_local -> p_local
# - Final classifier score (M3 output): fusion over branch logits
# - Losses: Masked BCE / Focal, Alignment Loss, Path Hinge Loss
#
# NOTE:
#   * This module outputs classifier-side scores; fusion with memory scores (M2) is done in M4.
#   * For cl loss you must provide positive label indices per sample.
#   * For path hinge loss you must provide parent–child edge list.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

    # Sample-sample alignment loss
    use_sample_cl: Optional[bool] = None
    inbatch_tau: float = 10.0
    weight_cl: float = 1.0

    # Path loss
    weight_path: Optional[float] = None
    use_path_loss: Optional[bool] = None
    path_on_local: Optional[bool] = None  # True -> use p_local; False -> use fused p_cls

    # Optional class weights
    pos_weight: Optional[Tensor] = None
    neg_weight: Optional[Tensor] = None

    def validate(self):
        required = {
            "use_sample_cl": self.use_sample_cl,
            "use_path_loss": self.use_path_loss,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "weight_path": self.weight_path,
            "path_on_local": self.path_on_local,
        }
        missing = [k for k, v in required.items() if v is None]
        if bool(self.use_sample_cl):
            cl_required = {
                "inbatch_tau": self.inbatch_tau,
            }
            missing.extend([k for k, v in cl_required.items() if v is None])
        if missing:
            raise ValueError(
                f"LossConfig missing required fields: {missing}. "
                "Provide them via TrainConfig/LossConfig to keep a single source of truth."
            )
        if self.inbatch_tau is not None and float(self.inbatch_tau) <= 0:
            raise ValueError("LossConfig.inbatch_tau must be > 0.")
        if self.weight_cl is None or float(self.weight_cl) < 0:
            raise ValueError("LossConfig.weight_cl must be >= 0.")


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
    local_head_hidden_ratio: Optional[float] = None
    local_dropout: Optional[float] = None
    # Fusion config (fixed to MLP)
    fusion_hidden_ratio: Optional[float] = None
    fusion_mode: str = "residual"

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
        if self.local_head_hidden_ratio is None:
            raise ValueError("ClassifierConfig.local_head_hidden_ratio must be provided.")
        if self.fusion_hidden_ratio is None:
            raise ValueError("ClassifierConfig.fusion_hidden_ratio must be provided.")
        if self.loss is None:
            raise ValueError("ClassifierConfig.loss must be provided (LossConfig).")
        self.fusion_mode = str(self.fusion_mode).strip().lower()
        valid_fusion_modes = {"direct_sum", "mlp_only", "residual"}
        if self.fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f"ClassifierConfig.fusion_mode must be one of {sorted(valid_fusion_modes)}; "
                f"got {self.fusion_mode!r}."
            )
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

            # Per-level prediction heads (all MLP)
            local_head_hidden = max(1, int(cfg.hidden_size * cfg.local_head_hidden_ratio))
            self.local_heads = nn.ModuleList()
            for n_l in self.level_sizes:
                head = nn.Sequential(
                    nn.Linear(d, local_head_hidden),
                    nn.GELU(),
                    nn.Dropout(local_drop),
                    nn.Linear(local_head_hidden, n_l),
                )
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

            # Level 1 base representation
            h_prev = self.local_first(h)                    # [B, d]

            # Single-vector attention (prev-only)
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
                logits_dtype = logits_local_list[0].dtype if logits_local_list else h.dtype
                p_local_dtype = p_local_list[0].dtype if p_local_list else h.dtype
                logits_local_concat = torch.zeros(h.size(0), self.L, device=h.device, dtype=logits_dtype)
                p_local_concat = torch.zeros(h.size(0), self.L, device=h.device, dtype=p_local_dtype)
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

        # Fusion MLP input is fixed to logits; fusion_mode selects how branch logits are combined.
        if self.use_local_branch and self.use_global_branch:
            if self.cfg.fusion_mode == "direct_sum":
                logits_sum = logits_global + logits_local_concat
            else:
                logits_concat = torch.cat([logits_local_concat, logits_global], dim=-1)  # [B, 2L], logits
                logits_fuse = self.fusion_mlp(logits_concat)
                if self.cfg.fusion_mode == "residual":
                    logits_sum = logits_global + logits_local_concat + logits_fuse
                else:
                    logits_sum = logits_fuse
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


def inbatch_loss_cl_loss(
    z: Tensor,
    y_true: Tensor,
    temperature: float = 10.0,
) -> Tensor:
    """
    Multi-label in-batch contrastive loss:
      C_ij = y_i . y_j
      beta_ij = C_ij / sum_{k!=i} C_ik
      L = -sum_i sum_{j!=i} beta_ij * log( exp(sim_ij/tau) / sum_{k!=i} exp(sim_ik/tau) )
    where sim_ij is cosine similarity.
    """
    if float(temperature) <= 0:
        raise ValueError("temperature must be > 0 for inbatch_loss_cl_loss.")

    B = z.size(0)
    if B <= 1:
        return z.sum() * 0.0

    y = (y_true > 0.5).to(dtype=z.dtype)
    sim = y @ y.T  # [B, B]
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, 0.0)

    sim_row_sum = sim.sum(dim=1, keepdim=True)
    beta = torch.where(
        sim_row_sum > 0,
        sim / sim_row_sum.clamp(min=1e-12),
        torch.zeros_like(sim),
    )

    z_norm = F.normalize(z, p=2, dim=-1)
    sim_zz = z_norm @ z_norm.T
    logits = sim_zz / float(temperature)
    logits = logits.masked_fill(eye, float("-inf"))

    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = logits - log_denom
    log_prob = torch.where(torch.isfinite(log_prob), log_prob, torch.zeros_like(log_prob))

    loss_matrix = -beta * log_prob
    return loss_matrix.sum()


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
    Align uses in-batch multi-label contrastive loss.
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
        # path resources
        edges_parent_child: List[Tuple[int, int]],
        # misc
        Y_sample_cl: Optional[Tensor] = None, # [B, L], optional terminal-positive labels for sample CL
        logits_cls: Optional[Tensor] = None,  # [B, L], optional for logit-space focal
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
        if getattr(self.loss, "use_sample_cl", True):
            y_cl = Y if Y_sample_cl is None else Y_sample_cl
            loss_cl = inbatch_loss_cl_loss(
                h_x,
                y_cl,
                temperature=float(getattr(self.loss, "inbatch_tau", 10.0)),
            )

        if getattr(self.loss, "use_path_loss", True) and getattr(self.loss, "weight_path", 0.0) != 0:
            loss_path = path_hinge_loss(p_cls, edges_parent_child)
        else:
            loss_path = torch.tensor(0.0, device=p_cls.device)

        total = loss_bce + float(getattr(self.loss, "weight_cl", 1.0)) * loss_cl + self.loss.weight_path * loss_path

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

    # Fake encoder outputs
    h_x = F.normalize(torch.randn(B, d, device=device), p=2, dim=-1)

    # Ground truth multi-hot Y and mask
    Y = torch.zeros(B, L, device=device)
    Y[0, [0, 4]] = 1
    Y[1, [1]] = 1
    Y[2, [2, 5]] = 1
    Y[3, [6]] = 1
    mask = torch.ones(B, L, device=device)

    # Parent-child edges (indices in concatenated label order)
    edges_pc = [(0, 3), (1, 4), (2, 5)]  # parents in L1, children in L2 (toy)

    cfg = ClassifierConfig(
        hidden_size=d,
        level_sizes=level_sizes,
        dropout=0.0,
        global_hidden_ratio=1.0,
        local_head_hidden_ratio=0.5,
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
            inbatch_tau=0.07,
            use_sample_cl=True,
            use_path_loss=True,
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
        edges_parent_child=edges_pc,
    )
    print({k: float(v) for k, v in losses.items()})
