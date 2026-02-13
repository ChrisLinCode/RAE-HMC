# memory.py
# RAE-HMC M2: Retrieval-Augmented Semantic Memory (static)
# - Keys K = [X; Z], Values V = [rho Y; (1 - rho) I_L]
# - Retrieval backend: FAISS IndexFlatIP (cosine under normalized embeddings)
# - Save/load with dtype-safe JSON.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import os, json
import numpy as np
import torch
import torch.nn.functional as F
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

Tensor = torch.Tensor

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class MemoryConfig:
    # Retrieval params
    top_b: int                  # truncated neighborhood size b
    tau_mem: float              # retrieval temperature
    rho: float                  # rho in V = [rho Y; (1 - rho) I_L]

    # Device & dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Normalize inputs if upstream did not already normalize them.
    assume_normalized: bool = True

    # Storage
    workdir: str = "./memory_store"

    # Retrieval backend: "faiss_ip" (legacy names are auto-mapped)
    backend: str = "faiss_ip"

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
# Semantic Memory
# -----------------------------
class SemanticMemory:
    """
    Implements the RAE-HMC memory with FAISS IP retrieval backend.
        K = [X; Z] where X in R^{N x d}, Z in R^{Lz x d}
        V = [rho Y; (1 - rho) I_L] where Y in {0,1}^{N x L}
    Provides: build / query / batch_query / save / load
    """

    def __init__(self, cfg: MemoryConfig):
        backend = str(getattr(cfg, "backend", "faiss_ip")).lower().strip()
        if backend in {"brute", "faiss_l2"}:
            backend = "faiss_ip"
        if backend != "faiss_ip":
            raise ValueError("MemoryConfig.backend must be 'faiss_ip'.")
        if faiss is None:
            raise ImportError(
                "FAISS backend requested (backend='faiss_ip') but faiss is not installed. "
                "Install with: pip install faiss-cpu"
            )
        cfg.backend = backend
        self.cfg = cfg

        # CPU tensors for persistence; GPU caches for fast math
        self.K_cpu: Optional[Tensor] = None  # [N+L, d]
        self.V_cpu: Optional[Tensor] = None  # [N+L, L]
        self.K_gpu: Optional[Tensor] = None
        self.V_gpu: Optional[Tensor] = None
        self.faiss_index = None

        self.N: int = 0
        self.L: int = 0
        self.d: int = 0
        self._built: bool = False
        self._scale: float = 1.0

    # ---------- Build ----------
    def build(self, X: Tensor, Z: Tensor, Y: Tensor, rho: Optional[float] = None, Z_label_ids: Optional[Tensor] = None) -> None:
        """
        Args:
            X: [N, d] text embeddings
            Z: [Lz, d] label embeddings
            Y: [N, L] multi-hot matrix
            Z_label_ids: [Lz] label id per Z row (required if Lz != L)
        """
        cfg = self.cfg
        dev = torch.device(cfg.device)

        if X.ndim != 2 or Z.ndim != 2: raise ValueError("X,Z must be 2D.")
        if Y.ndim != 2: raise ValueError("Y must be 2D.")
        N, d1 = X.shape; Lz, d2 = Z.shape
        if d1 != d2: raise ValueError(f"Embedding dim mismatch: X:{d1} vs Z:{d2}")
        L = int(Y.shape[1])
        if Y.shape != (N, L): raise ValueError(f"Y must be [N, L]; got {tuple(Y.shape)}")

        self.N, self.L, self.d = N, L, d1
        rho_val = cfg.rho if rho is None else float(rho)
        self._scale = max(rho_val, 1.0 - rho_val)

        if not cfg.assume_normalized:
            X = F.normalize(X, p=2, dim=-1)
            Z = F.normalize(Z, p=2, dim=-1)

        # CPU tensors for storage
        K = torch.cat([X, Z], dim=0).to(torch.float32).cpu()            # [N+Lz, d]
        I_L = torch.eye(L, dtype=torch.float32)
        rho_val = max(0.0, min(1.0, rho_val))  # clamp to [0,1]
        if Z_label_ids is None:
            if Lz != L:
                raise ValueError(f"Z rows ({Lz}) must equal num_labels ({L}) when Z_label_ids is None.")
            Z_label_ids = torch.arange(L, dtype=torch.long)
        if Z_label_ids.numel() != Lz:
            raise ValueError(f"Z_label_ids must have length {Lz}; got {Z_label_ids.numel()}.")
        V_z = (1.0 - rho_val) * I_L.index_select(0, Z_label_ids.to(I_L.device))
        V = torch.cat([rho_val * Y.to(torch.float32), V_z], dim=0).cpu()  # [N+Lz, L]

        self.K_cpu, self.V_cpu = K, V
        self._refresh_gpu_caches(dev)
        if self.cfg.backend == "faiss_ip":
            self._build_faiss_index()
        self._built = True

    def _refresh_gpu_caches(self, device: torch.device) -> None:
        if self.K_cpu is None or self.V_cpu is None: return
        self.K_gpu = self.K_cpu.to(device=device, dtype=self.cfg.dtype, non_blocking=True)
        self.V_gpu = self.V_cpu.to(device=device, dtype=self.cfg.dtype, non_blocking=True)

    def _build_faiss_index(self) -> None:
        if self.cfg.backend != "faiss_ip":
            self.faiss_index = None
            return
        if faiss is None:
            raise ImportError(
                "FAISS backend requested (backend='faiss_ip') but faiss is not installed. "
                "Install with: pip install faiss-cpu"
            )
        if self.K_cpu is None:
            raise RuntimeError("Cannot build FAISS index before memory keys are initialized.")
        K_np = np.ascontiguousarray(self.K_cpu.numpy().astype(np.float32, copy=False))
        index = faiss.IndexFlatIP(int(self.d))
        index.add(K_np)
        self.faiss_index = index

    # ---------- Retrieval ----------
    def query(
        self,
        q: Tensor,
        top_b: Optional[int] = None,
        tau_mem: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tensor:
        if q.ndim != 1 or (self.d and q.shape[0] != self.d):
            raise ValueError(f"q must be [d={self.d}]")
        tau_val = tau_mem if tau_mem is not None else temperature
        return self.batch_query(q.unsqueeze(0), top_b=top_b, tau_mem=tau_val)[0]

    def batch_query(
        self,
        Q: Tensor,
        top_b: Optional[int] = None,
        tau_mem: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tensor:
        if not self._built:
            raise RuntimeError("Memory not built. Call build() or load() first.")
        cfg = self.cfg
        dev = torch.device(cfg.device)
        B, d = Q.shape
        if d != self.d:
            raise ValueError(f"Q dim mismatch: got d={d}, expected d={self.d}")
        if self.K_cpu is None or self.V_cpu is None:
            raise RuntimeError("Memory tensors are not initialized. Call build() or load() first.")

        b = int(cfg.top_b if top_b is None else top_b)
        b = min(b, int(self.K_cpu.shape[0]))     # guard
        tau_val = tau_mem if tau_mem is not None else temperature
        tau = max(1e-6, float(cfg.tau_mem if tau_val is None else tau_val))
        if b <= 0:
            raise ValueError("top_b must be >= 1.")

        if self.faiss_index is None:
            self._build_faiss_index()
        Q_np = np.ascontiguousarray(Q.detach().to(torch.float32).cpu().numpy())
        sim_np, nn_idx_np = self.faiss_index.search(Q_np, b)
        sims = torch.from_numpy(sim_np).to(device=dev, dtype=self.cfg.dtype)
        nn_idx = torch.from_numpy(nn_idx_np.astype(np.int64, copy=False)).to(device=dev, dtype=torch.long)

        logits = sims / tau
        alpha = torch.softmax(logits, dim=-1)    # [B,b]

        if self.V_gpu is None:
            self._refresh_gpu_caches(dev)
        if self.V_gpu is None:
            raise RuntimeError("GPU value cache is not initialized. Call build() or load() first.")
        V_neighbors = self.V_gpu.index_select(0, nn_idx.view(-1)).view(B, b, self.L)
        S_mem = torch.bmm(alpha.unsqueeze(1), V_neighbors).squeeze(1)  # [B,L]
        scale = max(self._scale, 1e-6)
        S_mem = (S_mem / scale).clamp(min=0.0, max=1.0)
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
        if "rho" not in cfg_dict and "lambda_label" in cfg_dict:
            cfg_dict["rho"] = cfg_dict.pop("lambda_label")
        if "tau_mem" not in cfg_dict and "temperature" in cfg_dict:
            cfg_dict["tau_mem"] = cfg_dict.pop("temperature")
        elif "temperature" in cfg_dict:
            cfg_dict.pop("temperature", None)
        cfg_dict["dtype"] = _str_to_dtype(cfg_dict.get("dtype", "float32"))
        backend = str(cfg_dict.get("backend", "faiss_ip")).lower().strip()
        if backend in {"brute", "faiss_l2"}:
            backend = "faiss_ip"
        cfg_dict["backend"] = backend
        cfg = MemoryConfig(**cfg_dict)

        mem = cls(cfg)
        mem.K_cpu = torch.load(os.path.join(dirname, "K.pt"), map_location="cpu")
        mem.V_cpu = torch.load(os.path.join(dirname, "V.pt"), map_location="cpu")
        mem.N = int(meta["N"]); mem.L = int(meta["L"]); mem.d = int(meta["d"])
        mem._refresh_gpu_caches(torch.device(cfg.device))
        if cfg.backend == "faiss_ip":
            mem._build_faiss_index()
        mem._built = True
        return mem

    # kept for API compatibility; no-op in current backends
    def set_ef_search(self, ef: int) -> None:
        return

# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MemoryConfig(top_b=5, tau_mem=0.07, rho=1.0, assume_normalized=True, backend="faiss_ip")
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
    print("Reload ?:", float((S - S2).abs().max()))

