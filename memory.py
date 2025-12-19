# memory.py
# RAE-HMC M2 (Brute-force only): Retrieval-Augmented Semantic Memory (static)
# - Keys K = [X; Z], Values V = [λY; (1−λ) I_L]
# - Retrieval = exact cosine similarities via Q @ K^T + top-k + truncated softmax
# - No external dependencies. Save/load with dtype-safe JSON.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import os, json
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class MemoryConfig:
    # Retrieval params
    top_b: int                  # truncated neighborhood size b
    temperature: float          # τ_r
    lambda_label: float         # λ in V = [λY; (1−λ) I_L]

    # Device & dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Normalize inputs if upstream didn’t
    assume_normalized: bool = True

    # Storage
    workdir: str = "./memory_store"

    # Kept only for compatibility with previous scripts (ignored except 'brute')
    backend: str = "brute"      # must be "brute" in this file

# ---- dtype <-> str helpers for JSON ----
def _dtype_to_str(dt):
    if isinstance(dt, torch.dtype):
        return str(dt).replace("torch.", "")
    if isinstance(dt, str):
        return dt.replace("torch.", "")
    return "float32"

def _str_to_dtype(s):
    if isinstance(s, torch.dtype):
        return s
    if isinstance(s, str):
        name = s.replace("torch.", "")
        return getattr(torch, name, torch.float32)
    return torch.float32

# -----------------------------
# Semantic Memory (brute-force)
# -----------------------------
class SemanticMemory:
    """
    Implements the RAE-HMC memory (§3.4) with exact cosine retrieval.
        K = [X; Z]              where X ∈ ℝ^{N×d}, Z ∈ ℝ^{L×d}
        V = [λY; (1−λ) I_L]     where Y ∈ {0,1}^{N×L}
    Provides: build / query / batch_query / save / load
    """

    def __init__(self, cfg: MemoryConfig):
        if cfg.backend != "brute":
            raise ValueError("This memory.py is brute-force only. Set backend='brute'.")
        self.cfg = cfg

        # CPU tensors for persistence; GPU caches for fast math
        self.K_cpu: Optional[Tensor] = None  # [N+L, d]
        self.V_cpu: Optional[Tensor] = None  # [N+L, L]
        self.K_gpu: Optional[Tensor] = None
        self.V_gpu: Optional[Tensor] = None

        self.N: int = 0
        self.L: int = 0
        self.d: int = 0
        self._built: bool = False

    # ---------- Build ----------
    def build(self, X: Tensor, Z: Tensor, Y: Tensor, lambda_label: Optional[float] = None) -> None:
        """
        Args:
            X: [N, d] text embeddings
            Z: [L, d] label embeddings
            Y: [N, L] multi-hot matrix
        """
        cfg = self.cfg
        dev = torch.device(cfg.device)

        if X.ndim != 2 or Z.ndim != 2: raise ValueError("X,Z must be 2D.")
        if Y.ndim != 2: raise ValueError("Y must be 2D.")
        N, d1 = X.shape; L, d2 = Z.shape
        if d1 != d2: raise ValueError(f"Embedding dim mismatch: X:{d1} vs Z:{d2}")
        if Y.shape != (N, L): raise ValueError(f"Y must be [N, L]; got {tuple(Y.shape)}")

        self.N, self.L, self.d = N, L, d1
        lam = cfg.lambda_label if lambda_label is None else float(lambda_label)

        if not cfg.assume_normalized:
            X = F.normalize(X, p=2, dim=-1)
            Z = F.normalize(Z, p=2, dim=-1)

        # CPU tensors for storage
        K = torch.cat([X, Z], dim=0).to(torch.float32).cpu()            # [N+L, d]
        I_L = torch.eye(L, dtype=torch.float32)
        lam = max(0.0, min(1.0, lam))  # clamp to [0,1]
        V = torch.cat([lam * Y.to(torch.float32), (1.0 - lam) * I_L], dim=0).cpu()  # [N+L, L]

        self.K_cpu, self.V_cpu = K, V
        self._refresh_gpu_caches(dev)
        self._built = True

    def _refresh_gpu_caches(self, device: torch.device) -> None:
        if self.K_cpu is None or self.V_cpu is None: return
        self.K_gpu = self.K_cpu.to(device=device, dtype=self.cfg.dtype, non_blocking=True)
        self.V_gpu = self.V_cpu.to(device=device, dtype=self.cfg.dtype, non_blocking=True)

    # ---------- Retrieval ----------
    def query(self, q: Tensor, top_b: Optional[int] = None, temperature: Optional[float] = None) -> Tensor:
        if q.ndim != 1 or (self.d and q.shape[0] != self.d):
            raise ValueError(f"q must be [d={self.d}]")
        return self.batch_query(q.unsqueeze(0), top_b=top_b, temperature=temperature)[0]

    def batch_query(self, Q: Tensor, top_b: Optional[int] = None, temperature: Optional[float] = None) -> Tensor:
        if not self._built or self.K_gpu is None or self.V_gpu is None:
            raise RuntimeError("Memory not built. Call build() or load() first.")
        cfg = self.cfg
        dev = torch.device(cfg.device)
        B, d = Q.shape
        if d != self.d:
            raise ValueError(f"Q dim mismatch: got d={d}, expected d={self.d}")

        b = int(cfg.top_b if top_b is None else top_b)
        b = min(b, int(self.K_cpu.shape[0]))     # guard
        tau = max(1e-6, float(cfg.temperature if temperature is None else temperature))

        # exact cosine similarities via matrix multiply
        sims_full = (Q.to(dev) @ self.K_gpu.T)   # [B, N+L]
        sims, nn_idx = torch.topk(sims_full, k=b, dim=1, largest=True, sorted=False)  # [B,b]

        logits = sims / tau
        alpha = torch.softmax(logits, dim=-1)    # [B,b]

        V_neighbors = self.V_gpu.index_select(0, nn_idx.view(-1)).view(B, b, self.L)
        S_mem = torch.bmm(alpha.unsqueeze(1), V_neighbors).squeeze(1).clamp(min=0.0)  # [B,L]
        return S_mem

    # ---------- Persistence ----------
    def save(self, dirname: Optional[str] = None) -> None:
        if not self._built: raise RuntimeError("Nothing to save. Build or load first.")
        dirname = dirname or self.cfg.workdir
        os.makedirs(dirname, exist_ok=True)

        meta_cfg = asdict(self.cfg)
        meta_cfg["dtype"] = _dtype_to_str(meta_cfg.get("dtype", "float32"))
        meta = {"N": self.N, "L": self.L, "d": self.d, "cfg": meta_cfg}

        with open(os.path.join(dirname, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        torch.save(self.K_cpu, os.path.join(dirname, "K.pt"))
        torch.save(self.V_cpu, os.path.join(dirname, "V.pt"))

    @classmethod
    def load(cls, dirname: Optional[str] = None) -> "SemanticMemory":
        dirname = dirname or "./memory_store"
        with open(os.path.join(dirname, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg_dict = meta.get("cfg", {})
        cfg_dict["dtype"] = _str_to_dtype(cfg_dict.get("dtype", "float32"))
        # force brute to avoid mismatches
        cfg_dict["backend"] = "brute"
        cfg = MemoryConfig(**cfg_dict)

        mem = cls(cfg)
        mem.K_cpu = torch.load(os.path.join(dirname, "K.pt"), map_location="cpu")
        mem.V_cpu = torch.load(os.path.join(dirname, "V.pt"), map_location="cpu")
        mem.N = int(meta["N"]); mem.L = int(meta["L"]); mem.d = int(meta["d"])
        mem._refresh_gpu_caches(torch.device(cfg.device))
        mem._built = True
        return mem

    # kept for API compatibility; no-op in brute mode
    def set_ef_search(self, ef: int) -> None:
        return

# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MemoryConfig(top_b=5, temperature=0.07, lambda_label=1.0, assume_normalized=True, backend="brute")
    N, L, d = 6, 4, 8
    X = F.normalize(torch.randn(N, d), p=2, dim=-1)
    Z = F.normalize(torch.randn(L, d), p=2, dim=-1)
    Y = torch.zeros(N, L); Y[0,[0,1]]=1; Y[1,[1]]=1; Y[2,[2]]=1; Y[3,[3]]=1; Y[4,[0,2]]=1; Y[5,[1,3]]=1

    mem = SemanticMemory(cfg); mem.build(X, Z, Y)
    Q = F.normalize(torch.randn(3, d), p=2, dim=-1)
    S = mem.batch_query(Q); print("S_mem:", S.shape)
    s1 = mem.query(Q[0]);  print("s1:", s1.shape)

    mem.save("./memory_store")
    mem2 = SemanticMemory.load("./memory_store")
    S2 = mem2.batch_query(Q)
    print("Reload Δ:", float((S - S2).abs().max()))
