# inference.py
# RAE-HMC M4: Score Fusion & Thresholding (aligned with thesis §3.6)
# - s_final = γ * s_mem + (1 - γ) * s_cls
# - Binarization with a single global threshold θ (no additional closure)
# - Batch inference utilities, optional top-k selection, and ef_search control

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class InferenceConfig:
    gamma: float = 0.5           # fusion coefficient γ
    threshold: float = 0.5       # global decision threshold θ
    topk: Optional[int] = None   # optional: keep top-k before/after closure (applied before closure if set)
    use_logits_for_topk: bool = False  # if True, top-k uses logits-like space; here scores already in [0,1]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Hierarchy helpers
# -----------------------------
class Hierarchy:
    """
    Stores hierarchy relations required for closure.
    Provide either:
      - ancestors: Dict[int, List[int]]  (each node -> list of ancestors, excluding itself)
    Optionally:
      - parents: Dict[int, int] or Dict[int, List[int]] if multi-parent (not required for closure with ancestors)
    """
    def __init__(
        self,
        num_labels: int,
        ancestors: Dict[int, List[int]],
    ):
        self.L = num_labels
        self.ancestors = ancestors
        # Normalize to sorted unique lists
        for c, ancs in list(self.ancestors.items()):
            self.ancestors[c] = sorted(set(int(a) for a in ancs))

    @classmethod
    def from_edges(cls, num_labels: int, edges_parent_child: List[Tuple[int, int]]) -> "Hierarchy":
        """
        Build ancestor lists from parent-child edges on 0..L-1 label ids.
        """
        parents: Dict[int, List[int]] = {i: [] for i in range(num_labels)}
        children: Dict[int, List[int]] = {i: [] for i in range(num_labels)}
        for p, c in edges_parent_child:
            parents.setdefault(c, []).append(p)
            children.setdefault(p, []).append(c)

        # Compute ancestors by DFS from each node
        ancestors = {i: [] for i in range(num_labels)}
        for node in range(num_labels):
            stack = parents.get(node, [])[:]
            visited = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                ancestors[node].append(cur)
                stack.extend(parents.get(cur, []))
        return cls(num_labels, ancestors)


# -----------------------------
# Fusion & Closure
# -----------------------------
class InferenceEngine:
    """
    Implements §3.6 fusion and binarization.
    Inputs at inference:
      - s_mem: [B, L] memory scores (from M2)
      - p_cls: [B, L] classifier probabilities (from M3)
    Outputs:
      - s_final: [B, L] fused scores
      - y_hat:   [B, L] binary predictions after thresholding
    """

    def __init__(self, cfg: InferenceConfig, hierarchy: Hierarchy):
        self.cfg = cfg
        self.h = hierarchy

    # --- Fusion ---
    def fuse_scores(self, s_mem: Tensor, p_cls: Tensor, gamma: Optional[float] = None) -> Tensor:
        """
        s = γ * s_mem + (1 - γ) * p_cls, scores in [0,1].
        """
        g = self.cfg.gamma if gamma is None else float(gamma)
        # Clamp to [0,1] for safety then fuse
        s_mem = s_mem.clamp(0.0, 1.0)
        p_cls = p_cls.clamp(0.0, 1.0)
        s = g * s_mem + (1.0 - g) * p_cls
        return s

    # --- Binarization ---
    def binarize(self, scores: Tensor, threshold: Optional[float] = None, topk: Optional[int] = None) -> Tensor:
        """
        Apply optional top-k (on scores) then global threshold θ to produce {0,1}.
        If topk is set: keep top-k per sample as candidates (set others to 0), then apply threshold on the kept ones.
        """
        theta = self.cfg.threshold if threshold is None else float(threshold)
        B, L = scores.shape
        y = torch.zeros_like(scores, dtype=torch.int64)

        if topk is not None:
            k = max(1, int(topk))
            # top-k per row
            vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
            # Build a mask for kept positions
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            # Apply threshold only on kept positions
            kept_scores = scores * keep
            y = (kept_scores >= theta).to(torch.int64)
        else:
            y = (scores >= theta).to(torch.int64)
        return y

    # --- Full pipeline for a batch ---
    def predict_batch(
        self,
        s_mem: Tensor,   # [B, L]
        p_cls: Tensor,   # [B, L]
        gamma: Optional[float] = None,
        threshold: Optional[float] = None,
        topk: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Dict[str, Tensor]:
        """
        Returns dict with binarized predictions; optionally also fused scores.
        """
        s = self.fuse_scores(s_mem, p_cls, gamma=gamma)
        y = self.binarize(s, threshold=threshold, topk=topk or self.cfg.topk)
        if return_intermediate:
            return {"s_final": s, "y": y}
        return {"y": y}

    # Convenience for single sample
    def predict_single(
        self, s_mem_1: Tensor, p_cls_1: Tensor, gamma: Optional[float] = None,
        threshold: Optional[float] = None, topk: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Inputs are 1D [L], return 1D predictions.
        """
        s_mem = s_mem_1.view(1, -1)
        p_cls = p_cls_1.view(1, -1)
        out = self.predict_batch(s_mem, p_cls, gamma=gamma, threshold=threshold, topk=topk, return_intermediate=True)
        return {k: v[0] for k, v in out.items()}


# -----------------------------
# Optional: ef_search control hook for memory (M2)
# -----------------------------
def set_memory_ef_search(memory_obj, ef: int) -> None:
    """
    Wrapper to adjust ef_search at inference time, if the memory module exposes `set_ef_search`.
    """
    if hasattr(memory_obj, "set_ef_search"):
        memory_obj.set_ef_search(ef)


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Toy shapes
    B, L = 3, 7
    # Fake scores in [0,1]
    s_mem = torch.rand(B, L, device=device)
    p_cls = torch.rand(B, L, device=device)

    # A tiny hierarchy: level-0 parents: 0,1,2 ; level-1 children: 3..6
    # (0->3), (1->4), (1->5), (2->6)
    edges = [(0, 3), (1, 4), (1, 5), (2, 6)]
    H = Hierarchy.from_edges(num_labels=L, edges_parent_child=edges)

    cfg = InferenceConfig(gamma=0.6, threshold=0.5, topk=None, device=device)
    engine = InferenceEngine(cfg, H)

    out = engine.predict_batch(s_mem, p_cls, return_intermediate=True)
    print("s_final shape:", tuple(out["s_final"].shape))
    print("binary predictions:", out["y"])
