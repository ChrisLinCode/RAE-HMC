# train_rae_hmc.py
# End-to-end training + validation + test for RAE-HMC
# Splits train/val/test from a single dataset.csv and prints research metrics to console.
# Modules required in same folder: build_hierarchy_utils.py, encoder.py, memory.py, classifier.py, inference.py

import os, json, random, math
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from build_hierarchy_utils import (
    load_hierarchy_from_file, build_multi_hot_Y, make_level_slices
)
from encoder import SharedEncoder, EncoderConfig
from memory import SemanticMemory, MemoryConfig
from classifier import ClassifierConfig, DualBranchHierClassifier, JointLossCombiner, LossConfig
from inference import InferenceConfig, InferenceEngine, Hierarchy


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    dataset_csv: str = "dataset/dataset.csv"
    hierarchy_json: str = "dataset/label_hierarchy.json"
    text_col: str = "text"
    labels_col: str = "labels"   # labels separated by ';' or ','

    # Split ratios (train + val + test should be 1.0)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # Encoder
    model_name: str = "bert-base-chinese"
    max_len: int = 32
    batch_size: int = 16
    # Two-stage training hyperparameters
    contrast_epochs: int = 3              # stage 1: contrastive/align pretrain epochs
    classifier_epochs: int = 12           # stage 2: classifier training epochs
    contrast_lr: float = 3e-5             # encoder lr for stage 1 (also used to finetune encoder in stage 2)
    classifier_lr: float = 3e-3          # classifier head lr for stage 2
    seed: int = 42
    warmup_ratio: float = 0.15

    # Memory (M2)
    top_b: int = 50
    top_b_candidates: List[int] = field(default_factory=lambda: [25, 50, 75, 100, 125, 150])
    temperature: float = 0.04 #論文配置
    rho: float = 0.5
    rho_candidates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    memory_build_mode: str = "prototype"  # "sample" or "prototype"
    memory_proto_k: int = 1
    memory_proto_max_iters: int = 10
    memory_proto_min_samples: int = 2 #k=1時沒作用

    # Classifier (M3)
    global_head_mode: str = "mlp"  # "linear" or "mlp"
    global_hidden_ratio: float = 1.25
    dropout: float = 0.15
    focal_alpha: float = 0.5
    focal_gamma: float = 0.0

    # path loss
    use_path_loss: bool = True
    weight_path: float = 0.5

    # label loss
    use_label_loss: bool = True
    weight_label: float = 0.15
    num_neg_label: int = 20
    tau_label: float = 0.07

    # align loss
    use_align_loss: bool = True
    weight_align: float = 0.25  #0.25
    num_neg_align: int = 10
    tau_align: float = 0.04 #論文配置

    # Fusion (M4)
    eta: float = 0.5
    delta: float = 0.5
    topk: Optional[int] = 15
    eta_candidates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    delta_candidates: List[float] = field(default_factory=lambda: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    val_metric: str = "macro"  # metric for selecting best validation parameters ("micro" or "macro")

    # Module switches / ablations
    use_memory: bool = True
    use_local_branch: bool = True   
    use_global_branch: bool = True
    run_ablation: bool = False     # If True, run predefined ablation scenarios
    run_cl_ablation: bool = False  # If True, run CL-loss ablations (align/label)

    # Sampling (tail-aware / level-aware)
    # Four-bin tail weighting by fixed label frequency quartiles (0-25/25-50/50-75/75-100)
    tail_weight_q0_25: float = 1.75
    tail_weight_q25_50: float = 1.5
    tail_weight_q50_75: float = 1.25
    tail_weight_q75_100: float = 1.0
    level_threshold: int = 4
    level_weight: float = 0.25
    weighted_extra_ratio_stage1: float = 1.0
    weighted_extra_ratio_stage2: float = 1.0

    # Output dir
    workdir: str = "./outputs"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_label_cell(cell: str) -> List[str]:
    if pd.isna(cell): return []
    s = str(cell)
    if ";" in s: return [t.strip() for t in s.split(";") if t.strip()]
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s.strip()] if s.strip() else []

def split_train_val_test(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1.0"
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    tr_idx  = idx[:n_train]
    va_idx  = idx[n_train:n_train + n_val]
    te_idx  = idx[n_train + n_val:]
    return (df.iloc[tr_idx].reset_index(drop=True),
            df.iloc[va_idx].reset_index(drop=True),
            df.iloc[te_idx].reset_index(drop=True))

def iterative_stratified_split(
    Y: np.ndarray,
    test_size: float,
    seed: int,
    ensure_test_label_coverage: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple iterative stratification for multi-label data.
    Returns (train_indices, test_indices) given desired test_size ratio.
    If ensure_test_label_coverage=True, pre-select one sample per label (if available)
    into the test set before running the iterative procedure.
    """
    rng = np.random.default_rng(seed)
    n_samples = Y.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    test_count = max(1, int(round(n_samples * test_size)))
    available = np.ones(n_samples, dtype=bool)
    label_totals = Y.sum(axis=0).astype(float)
    desired = label_totals * (test_count / max(1, n_samples))
    assigned = np.zeros_like(desired)
    test_indices: List[int] = []

    if ensure_test_label_coverage:
        # Pre-pick one sample per label (if any) to ensure the test set sees it.
        L = Y.shape[1]
        for label in range(L):
            candidates = np.where((Y[:, label] > 0) & available)[0]
            if len(candidates) == 0:
                continue
            idx = int(rng.choice(candidates))
            available[idx] = False
            test_indices.append(idx)
            assigned += Y[idx]
        # Increase target size if coverage used more than planned quota
        test_count = max(test_count, len(test_indices))

    while len(test_indices) < test_count and available.any():
        need = desired - assigned
        if np.all(need <= 0):
            candidates = np.where(available)[0]
            if len(candidates) == 0:
                break
            idx = int(rng.choice(candidates))
        else:
            need_mask = need.copy()
            need_mask[need_mask <= 0] = -np.inf
            label = int(np.argmax(need_mask))
            candidates = np.where((Y[:, label] > 0) & available)[0]
            if len(candidates) == 0:
                candidates = np.where(available)[0]
                if len(candidates) == 0:
                    break
            idx = int(rng.choice(candidates))
        available[idx] = False
        test_indices.append(idx)
        assigned += Y[idx]

    train_indices = np.where(available)[0]
    return train_indices.astype(int), np.array(test_indices, dtype=int)

def subset_tokens(tokens: Dict[str, torch.Tensor], indices: np.ndarray) -> Dict[str, torch.Tensor]:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    return {k: v.index_select(0, idx_tensor) for k, v in tokens.items()}

def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = (y_true & y_pred).sum()
    fp = ((~y_true.astype(bool)) & y_pred.astype(bool)).sum()
    fn = (y_true.astype(bool) & (~y_pred.astype(bool))).sum()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # per-label F1 averaged over labels
    L = y_true.shape[1]
    f1s = []
    for j in range(L):
        yt = y_true[:, j].astype(bool)
        yp = y_pred[:, j].astype(bool)
        tp = (yt & yp).sum()
        fp = ((~yt) & yp).sum()
        fn = (yt & (~yp)).sum()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def get_val_metric_name(cfg: TrainConfig) -> str:
    metric = str(getattr(cfg, "val_metric", "micro")).lower().strip()
    return "macro" if metric == "macro" else "micro"


def compute_val_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: TrainConfig
) -> Tuple[float, float, float]:
    micro = micro_f1(y_true, y_pred)
    macro = macro_f1(y_true, y_pred)
    metric = get_val_metric_name(cfg)
    score = macro if metric == "macro" else micro
    return score, micro, macro

def get_top_b_candidates(cfg: TrainConfig) -> List[int]:
    candidates = list(getattr(cfg, "top_b_candidates", None) or [cfg.top_b])
    if cfg.top_b not in candidates:
        candidates.append(cfg.top_b)
    cleaned: List[int] = []
    seen = set()
    for b in candidates:
        try:
            b_int = int(b)
        except (TypeError, ValueError):
            continue
        if b_int <= 0:
            continue
        if b_int not in seen:
            cleaned.append(b_int)
            seen.add(b_int)
    return cleaned if cleaned else [int(cfg.top_b)]

def per_label_report(y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict[int, str]):
    """
    Print per-label precision/recall/F1 and positive count for inspection (used at test time).
    """
    L = y_true.shape[1]
    for j in range(L):
        yt = y_true[:, j].astype(bool)
        yp = y_pred[:, j].astype(bool)
        support = int(yt.sum())
        tp = (yt & yp).sum()
        fp = ((~yt) & yp).sum()
        fn = (yt & (~yp)).sum()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        name = id2label.get(j, str(j))
        print(f"[Label {j:03d}] {name} | p={p:.4f} r={r:.4f} f1={f1:.4f} | support={support}")

def tokenize_texts(tokenizer, texts: List[str], max_length: int, chunk_size: int = 256) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, List[torch.Tensor]] = {}
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        encoded = tokenizer(
            chunk,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        for key, value in encoded.items():
            tensors.setdefault(key, []).append(value)
    return {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in tensors.items()}


def move_tokens_to_device(tokens: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in tokens.items()}


def slice_tokens(tokens: Dict[str, torch.Tensor], start: int, end: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v[start:end].to(device) for k, v in tokens.items()}

def predict_with_strategy(
    s_mem: torch.Tensor,
    p_cls: torch.Tensor,
    engine: InferenceEngine,
    cfg: TrainConfig,
    eta_override: Optional[float] = None,
    delta_override: Optional[float] = None,
) -> torch.Tensor:
    """
    Decide prediction path based on available modules.
    - If memory + fusion (auto when memory on): fuse s_mem and p_cls.
    - If memory only: apply delta cutoff to memory scores.
    - Else: apply delta cutoff to classifier scores (global or global+local depending on clf config).
    """
    eta = eta_override if eta_override is not None else cfg.eta
    delta = delta_override if delta_override is not None else cfg.delta
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if fusion_on:
        return engine.predict_batch(
            s_mem, p_cls, eta=eta, delta=delta
        )["y"]
    if cfg.use_memory:
        return (s_mem >= delta).to(torch.int64)
    return (p_cls >= delta).to(torch.int64)


def classifier_enabled(cfg: TrainConfig) -> bool:
    return bool(cfg.use_global_branch or cfg.use_local_branch)


def tune_memory_only_delta(
    s_mem_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[float, float, float, float]:
    """
    Tune a single global delta for memory-only predictions:
        y_hat = 1[s_mem >= delta]
    Returns (best_delta, best_score, best_micro_f1, best_macro_f1).
    """
    y_true = (Y_val.detach().cpu().numpy() > 0.5).astype(np.int32)
    candidates = list(getattr(cfg, "delta_candidates", None) or [cfg.delta])
    if cfg.delta not in candidates:
        candidates.append(cfg.delta)
    # Guard: keep within [0,1]
    candidates = [float(max(0.0, min(1.0, t))) for t in candidates]

    best_delta = float(cfg.delta)
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    s_cpu = s_mem_val.detach().cpu()
    for delta_val in candidates:
        y_pred = (s_cpu >= delta_val).numpy().astype(np.int32)
        score, micro, macro = compute_val_metrics(y_true, y_pred, cfg)
        if score > best_score:
            best_score = score
            best_micro = micro
            best_macro = macro
            best_delta = float(delta_val)
    return best_delta, best_score, best_micro, best_macro


def tune_classifier_only_delta(
    p_cls_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[float, float, float, float]:
    """
    Tune a single global delta for classifier-only predictions:
        y_hat = 1[p_cls >= delta]
    Returns (best_delta, best_score, best_micro_f1, best_macro_f1).
    """
    y_true = (Y_val.detach().cpu().numpy() > 0.5).astype(np.int32)
    candidates = list(getattr(cfg, "delta_candidates", None) or [cfg.delta])
    if cfg.delta not in candidates:
        candidates.append(cfg.delta)
    # Guard: keep within [0,1]
    candidates = [float(max(0.0, min(1.0, t))) for t in candidates]

    best_delta = float(cfg.delta)
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    p_cpu = p_cls_val.detach().cpu()
    for delta_val in candidates:
        y_pred = (p_cpu >= delta_val).numpy().astype(np.int32)
        score, micro, macro = compute_val_metrics(y_true, y_pred, cfg)
        if score > best_score:
            best_score = score
            best_micro = micro
            best_macro = macro
            best_delta = float(delta_val)
    return best_delta, best_score, best_micro, best_macro


def encode_with_encoder(
    encoder: SharedEncoder,
    tokens: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device
    ) -> torch.Tensor:
    encoder.eval()
    outputs = []
    N = tokens["input_ids"].size(0)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            batch = slice_tokens(tokens, s, e, device)
            out = encoder.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids")
            )
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)


def select_tokens_by_index(tokens: Dict[str, torch.Tensor], indices: List[int], device: torch.device) -> Dict[str, torch.Tensor]:
    # Place indices on the same device as token tensors to avoid device mismatch when tokens are on GPU.
    token_device = next(iter(tokens.values())).device
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=token_device)
    return {k: v.index_select(0, idx_tensor).to(device) for k, v in tokens.items()}


def to_pos_idx_list(Y: torch.Tensor, label_levels: List[int], min_level: int = 3) -> List[List[int]]:
    idxs: List[List[int]] = []
    Yn = (Y > 0.5).cpu().numpy()
    for row in Yn:
        all_indices = np.nonzero(row)[0].tolist()
        filtered = [j for j in all_indices if label_levels[j] >= min_level]
        idxs.append(filtered if filtered else all_indices)
    return idxs


def build_tail_level_masks(
    Y: torch.Tensor,
    level_threshold: int,
    label_levels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = Y.sum(dim=0).cpu().numpy()
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        q25 = 0.0
        q50 = 0.0
        q75 = 0.0
    else:
        q25 = float(np.percentile(nonzero, 25))
        q50 = float(np.percentile(nonzero, 50))
        q75 = float(np.percentile(nonzero, 75))
    tail_mask_q0_25 = torch.tensor(counts <= q25, dtype=torch.bool)
    tail_mask_q25_50 = torch.tensor((counts > q25) & (counts <= q50), dtype=torch.bool)
    tail_mask_q50_75 = torch.tensor((counts > q50) & (counts <= q75), dtype=torch.bool)
    tail_mask_q75_100 = torch.tensor(counts > q75, dtype=torch.bool)
    level_mask = torch.tensor([lvl >= level_threshold for lvl in label_levels], dtype=torch.bool)
    return tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100, level_mask


def compute_sample_weights(
    Y: torch.Tensor,
    tail_mask_q0_25: torch.Tensor,
    tail_mask_q25_50: torch.Tensor,
    tail_mask_q50_75: torch.Tensor,
    tail_mask_q75_100: torch.Tensor,
    level_mask: torch.Tensor,
    tail_weight_q0_25: float,
    tail_weight_q25_50: float,
    tail_weight_q50_75: float,
    tail_weight_q75_100: float,
    level_weight: float
) -> torch.Tensor:
    weights = torch.ones(Y.size(0), device=Y.device)
    if tail_mask_q75_100.any():
        tail_hits_q75_100 = (Y[:, tail_mask_q75_100.to(Y.device)] > 0.5).any(dim=1)
        if tail_hits_q75_100.any():
            weights = torch.where(
                tail_hits_q75_100,
                torch.tensor(tail_weight_q75_100, device=Y.device),
                weights
            )
    if tail_mask_q50_75.any():
        tail_hits_q50_75 = (Y[:, tail_mask_q50_75.to(Y.device)] > 0.5).any(dim=1)
        if tail_hits_q50_75.any():
            weights = torch.where(
                tail_hits_q50_75,
                torch.tensor(tail_weight_q50_75, device=Y.device),
                weights
            )
    if tail_mask_q25_50.any():
        tail_hits_q25_50 = (Y[:, tail_mask_q25_50.to(Y.device)] > 0.5).any(dim=1)
        if tail_hits_q25_50.any():
            weights = torch.where(
                tail_hits_q25_50,
                torch.tensor(tail_weight_q25_50, device=Y.device),
                weights
            )
    if tail_mask_q0_25.any():
        tail_hits_q0_25 = (Y[:, tail_mask_q0_25.to(Y.device)] > 0.5).any(dim=1)
        if tail_hits_q0_25.any():
            weights = torch.where(
                tail_hits_q0_25,
                torch.tensor(tail_weight_q0_25, device=Y.device),
                weights
            )
    if level_mask.any():
        level_hits = (Y[:, level_mask.to(Y.device)] > 0.5).any(dim=1)
        weights += level_weight * level_hits.float()
    return weights


def get_memory_build_mode(cfg: TrainConfig) -> str:
    mode = str(getattr(cfg, "memory_build_mode", "sample")).lower().strip()
    if mode in {"prototype", "proto", "centroid", "kmeans"}:
        return "prototype"
    return "sample"


def _kmeans_cosine_centroids(
    X: torch.Tensor,
    k: int,
    max_iters: int,
    seed: int
) -> torch.Tensor:
    n = X.size(0)
    if n == 0:
        raise ValueError("kmeans requires at least 1 sample.")
    if k <= 1:
        return F.normalize(X.mean(dim=0, keepdim=True), p=2, dim=1)
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    centroids = X.index_select(0, perm[:k]).clone()
    prev_assign = None
    for _ in range(max_iters):
        sims = X @ centroids.t()
        assign = sims.argmax(dim=1)
        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign
        new_centroids = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                c = X[mask].mean(dim=0)
            else:
                idx = int(torch.randint(0, n, (1,), generator=gen))
                c = X[idx]
            new_centroids.append(c)
        centroids = torch.stack(new_centroids, dim=0)
        centroids = F.normalize(centroids, p=2, dim=1)
    return centroids


def build_memory_prototypes(
    X: torch.Tensor,
    Y: torch.Tensor,
    hd,
    cfg: TrainConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_cpu = X.detach().cpu()
    Y_cpu = Y.detach().cpu()
    L = int(Y_cpu.size(1))
    k = max(1, int(getattr(cfg, "memory_proto_k", 1)))
    max_iters = max(1, int(getattr(cfg, "memory_proto_max_iters", 10)))
    min_samples = max(1, int(getattr(cfg, "memory_proto_min_samples", 2)))
    ancestors = getattr(hd, "ancestors", {})

    X_out: List[torch.Tensor] = []
    Y_out: List[torch.Tensor] = []

    for label in range(L):
        idx = torch.nonzero(Y_cpu[:, label] > 0.5, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        X_label = X_cpu.index_select(0, idx)
        if k <= 1 or idx.numel() < max(k, min_samples):
            centroids = F.normalize(X_label.mean(dim=0, keepdim=True), p=2, dim=1)
            k_eff = 1
        else:
            k_eff = min(k, int(idx.numel()))
            centroids = _kmeans_cosine_centroids(X_label, k_eff, max_iters, seed=cfg.seed + label)

        target = torch.zeros(L, dtype=torch.float32)
        target[label] = 1.0
        for anc in ancestors.get(label, []):
            target[int(anc)] = 1.0
        X_out.append(centroids)
        Y_out.append(target.unsqueeze(0).repeat(k_eff, 1))

    if not X_out:
        return X_cpu, Y_cpu
    return torch.cat(X_out, dim=0), torch.cat(Y_out, dim=0)


def prepare_memory_inputs(
    X: torch.Tensor,
    Y: torch.Tensor,
    cfg: TrainConfig,
    hd
) -> Tuple[torch.Tensor, torch.Tensor]:
    if get_memory_build_mode(cfg) == "prototype":
        return build_memory_prototypes(X, Y, hd, cfg)
    return X, Y


def tune_fusion_parameters(
    enc: SharedEncoder,
    clf: DualBranchHierClassifier,
    mem: SemanticMemory,
    engine: InferenceEngine,
    tokens: Dict[str, torch.Tensor],
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device
) -> Tuple[float, float, float, float, float, int]:
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if not fusion_on:
        return cfg.eta, cfg.delta, -1.0, -1.0, -1.0, cfg.top_b
    X_va_enc = encode_with_encoder(enc, tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"]
    y_true_va = (Y_val.cpu().numpy() > 0.5).astype(np.int32)
    best_eta = cfg.eta
    best_delta = cfg.delta
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    best_top_b = cfg.top_b
    top_b_candidates = get_top_b_candidates(cfg)
    for top_b in top_b_candidates:
        with torch.no_grad():
            s_mem_va = mem.batch_query(X_va_dev, top_b=top_b)
        for eta in cfg.eta_candidates:
            for delta_val in cfg.delta_candidates:
                pred = engine.predict_batch(
                    s_mem_va, p_cls_va, eta=eta, delta=delta_val
                )
                y_pred = pred["y"].cpu().numpy().astype(np.int32)
                score, micro, macro = compute_val_metrics(y_true_va, y_pred, cfg)
                if score > best_score:
                    best_score = score
                    best_micro = micro
                    best_macro = macro
                    best_eta = eta
                    best_delta = delta_val
                    best_top_b = int(top_b)
    return best_eta, best_delta, best_score, best_micro, best_macro, best_top_b



# -----------------------------
# Main
# -----------------------------
def main(cfg: TrainConfig, summary: Optional[List[Dict[str, float]]] = None, scenario_name: Optional[str] = None) -> Optional[Dict[str, float]]:
    if cfg.run_ablation or getattr(cfg, "run_cl_ablation", False):
        summary = summary if summary is not None else []
        scenarios: List[Tuple[str, Dict[str, object]]] = []

        if cfg.run_ablation:
            scenarios.extend([
                ("global_only", {"use_memory": False, "use_local_branch": False, "use_global_branch": True}),
                ("local_only", {"use_memory": False, "use_local_branch": True, "use_global_branch": False}),
                ("memory_only", {"use_memory": True, "use_local_branch": False, "use_global_branch": False}),
                ("global_local", {"use_memory": False, "use_local_branch": True, "use_global_branch": True}), 
                ("global+mem", {"use_memory": True, "use_local_branch": False, "use_global_branch": True}),
                ("local+mem", {"use_memory": True, "use_local_branch": True, "use_global_branch": False}),
                ("all", {"use_memory": True, "use_local_branch": True, "use_global_branch": True}),
            ])

        if getattr(cfg, "run_cl_ablation", False):
            align_w = cfg.weight_align
            label_w = cfg.weight_label
            scenarios.extend([
                ("cl_all", {}),
                ("cl_align_only", {
                    "use_align_loss": True, "weight_align": align_w,
                    "use_label_loss": False, "weight_label": 0.0,
                }),
                ("cl_label_only", {
                    "use_align_loss": False, "weight_align": 0.0,
                    "use_label_loss": True, "weight_label": label_w,
                }),
                ("cl_none", {
                    "use_align_loss": False, "weight_align": 0.0,
                    "use_label_loss": False, "weight_label": 0.0,
                }),
            ])

        for name, overrides in scenarios:
            sub_cfg = replace(
                cfg,
                **overrides,
                run_ablation=False,
                run_cl_ablation=False,
                workdir=os.path.join(cfg.workdir, name)
            )
            print(f"\n[Ablation] Running scenario: {name} with {overrides}")
            res = main(sub_cfg, summary=summary, scenario_name=name)
            if res is not None:
                summary.append(res)
        # Print summary table
        if summary:
            print("\n[Ablation Summary]")
            summary_sorted = summary
            if getattr(cfg, "run_cl_ablation", False):
                cl_order = [
                    "cl_none",
                    "cl_label_only",
                    "cl_align_only",
                    "cl_all",
                ]
                order_index = {name: i for i, name in enumerate(cl_order)}
                summary_sorted = sorted(
                    summary,
                    key=lambda item: order_index.get(item.get("scenario", ""), len(order_index))
                )
            for item in summary_sorted:
                use_memory = bool(item.get("use_memory", True))
                use_global = bool(item.get("use_global_branch", True))
                use_local = bool(item.get("use_local_branch", True))
                classifier_on = use_global or use_local
                fusion_on = use_memory and classifier_on

                eta_print = f"{float(item.get('eta', 0.0)):.2f}" if fusion_on else "N/A"
                delta_print = f"{float(item.get('delta', 0.0)):.2f}" if (use_memory or classifier_on) else "N/A"
                rho_val = item.get("rho", None)
                rho_print = f"{float(rho_val):.2f}" if (use_memory and rho_val is not None) else "N/A"
                top_b_val = item.get("top_b", None)
                top_b_print = f"{int(top_b_val)}" if (use_memory and top_b_val is not None) else "N/A"
                print(f"  - {item['scenario']}: eta={eta_print}, delta={delta_print}, rho={rho_print}, top_b={top_b_print}, "
                      f"micro-F1={item['micro']:.4f}, macro-F1(all)={item['macro_all']:.4f}")
        return None

    os.makedirs(cfg.workdir, exist_ok=True)
    set_seed(cfg.seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    #device_str = "cpu"
    device = torch.device(device_str)
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    print(f"Using device: {device}")

    # 1) Load hierarchy and dataset
    hd = load_hierarchy_from_file(cfg.hierarchy_json)
    L = hd.num_labels
    print(f"[Hierarchy] num_labels={L}, level_sizes={hd.level_sizes}")
    level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
    label_levels = [level_lookup.get(i, 1) for i in range(L)]
    same_level_map: Dict[int, List[int]] = {}
    for idx, lvl in enumerate(label_levels):
        same_level_map.setdefault(lvl, []).append(idx)
    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [parse_label_cell(s) for s in df_all[cfg.labels_col].tolist()]
    Y_all = np.array(build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
    train_idx_np, test_idx_np = iterative_stratified_split(
        Y_all, cfg.test_ratio, cfg.seed, ensure_test_label_coverage=True
    )
    df_train = df_all.iloc[train_idx_np].reset_index(drop=True)
    df_test = df_all.iloc[test_idx_np].reset_index(drop=True)
    print(f"[Data] Train={len(df_train)} | Test={len(df_test)} (stratified 4:1)")

    tr_texts = df_train[cfg.text_col].astype(str).tolist()
    te_texts = df_test[cfg.text_col].astype(str).tolist()

    # Build Y tensors for train/test sets
    Y_tr_full = torch.tensor(Y_all[train_idx_np], dtype=torch.float32)
    Y_te = torch.tensor(Y_all[test_idx_np], dtype=torch.float32)

    # 3) M1: Shared encoder (jointly trained)
    print("[Stage] Initializing shared encoder...")
    enc = SharedEncoder(EncoderConfig(model_name=cfg.model_name, max_length=cfg.max_len, pooling="cls", normalize=True, device=device_str))
    print("[Stage] Tokenizing train/test/label texts...")
    label_descs = [hd.path_strings[i] for i in range(hd.num_labels)]
    train_tokens = tokenize_texts(enc.tokenizer, tr_texts, cfg.max_len)
    test_tokens = tokenize_texts(enc.tokenizer, te_texts, cfg.max_len)
    label_tokens = tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
    del enc

    # 4) M3: Classifier + losses (M2 memory will be refreshed with current encoder later)
    level_slices = make_level_slices(hd.levels)
    edges_pc = [(int(p), int(c)) for (p, c) in hd.edges_parent_child]
    hierarchy_obj = Hierarchy(num_labels=L, ancestors=hd.ancestors)

    mem_cfg = MemoryConfig(
        backend="brute",
        top_b=cfg.top_b,
        temperature=cfg.temperature,
        rho=cfg.rho,
        device=device_str
    )

    # Prepare fold splits (4-fold CV, stratified via iterative_stratified_split per fold)
    def make_k_folds_indices(Y_np: np.ndarray, k: int, seed: int):
        n = len(Y_np)
        all_idx = np.arange(n)
        remaining = all_idx.copy()
        folds = []
        cur_seed = seed
        for i in range(k):
            if i == k - 1:
                val_idx = remaining
            else:
                Y_rem = Y_np[remaining]
                tr_rel, va_rel = iterative_stratified_split(Y_rem, test_size=1.0 / (k - i), seed=cur_seed, ensure_test_label_coverage=True)
                val_idx = remaining[va_rel]
                cur_seed += 1
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
            folds.append((f"fold{i+1}", train_idx, val_idx))
            remaining = np.setdiff1d(remaining, val_idx, assume_unique=False)
        return folds

    fold_splits = make_k_folds_indices(Y_tr_full.cpu().numpy(), k=2, seed=cfg.seed + 1)

    fold_results = []
    best_fold = None

    for fold_name, tr_idx_arr, va_idx_arr in fold_splits:
        result = train_single_fold(
            fold_name=fold_name,
            cfg=cfg,
            hd=hd,
            hierarchy_obj=hierarchy_obj,
            level_slices=level_slices,
            label_levels=label_levels,
            same_level_map=same_level_map,
            edges_pc=edges_pc,
            label_tokens=label_tokens,
            train_tokens_full=train_tokens,
            Y_train_full=Y_tr_full,
            train_indices=tr_idx_arr,
            val_indices=va_idx_arr,
            device=device,
            device_str=device_str,
            mem_cfg=mem_cfg
        )
        fold_results.append(result)
        if best_fold is None or result["best_val_score"] > best_fold["best_val_score"]:
            best_fold = result

    val_metric = get_val_metric_name(cfg)
    print("\n[CV] Fold summary:")
    for res in fold_results:
        use_memory = bool(res.get("use_memory", True))
        rho_info = res.get("last_tuned_rho", None)
        eta_info = res.get("last_tuned_eta", None)
        delta_info = res.get("last_tuned_delta", None)
        metric_val = float(res.get("best_val_score", -1.0))
        micro_val = float(res.get("best_val_micro", -1.0))
        macro_val = float(res.get("best_val_macro", -1.0))
        if use_memory and rho_info is not None:
            top_b_info = res.get("last_tuned_top_b", None)
            top_b_print = f", top_b={int(top_b_info)}" if top_b_info is not None else ""
            print(f"  - {res['fold']}: best val {val_metric}-F1={metric_val:.4f}, "
                  f"micro-F1={micro_val:.4f}, macro-F1={macro_val:.4f}, "
                  f"rho={rho_info:.2f}, eta={eta_info:.2f}, delta={delta_info:.2f}{top_b_print}, checkpoint={res['best_path']}")
        else:
            print(f"  - {res['fold']}: best val {val_metric}-F1={metric_val:.4f}, "
                  f"micro-F1={micro_val:.4f}, macro-F1={macro_val:.4f}, checkpoint={res['best_path']}")

    if best_fold is None:
        print("No folds completed. Exiting.")
        return

    print(f"\n[CV] Selecting {best_fold['fold']} hyperparameters for final training.")
    # Clean up non-selected fold checkpoints
    for res in fold_results:
        path = res.get("best_path")
        if path and os.path.exists(path) and path != best_fold.get("best_path"):
            try:
                os.remove(path)
                print(f"[Cleanup] Removed fold checkpoint: {path}")
            except OSError as e:
                print(f"[Cleanup] Failed to remove {path}: {e}")

    eta_final = best_fold["last_tuned_eta"] if best_fold.get("last_tuned_eta") is not None else cfg.eta
    delta_final = best_fold["last_tuned_delta"] if best_fold.get("last_tuned_delta") is not None else cfg.delta
    rho_final = best_fold["last_tuned_rho"] if best_fold.get("last_tuned_rho") is not None else cfg.rho
    top_b_final = best_fold["last_tuned_top_b"] if best_fold.get("last_tuned_top_b") is not None else cfg.top_b
    mem_cfg_final = replace(mem_cfg, rho=rho_final, top_b=top_b_final)

    # ----------------- Retrain on full training data -----------------
    enc, clf, clf_cfg_full = train_full_model(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        same_level_map=same_level_map,
        edges_pc=edges_pc,
        label_tokens=label_tokens,
        train_tokens_full=train_tokens,
        Y_train_full=Y_tr_full,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg_final
    )

    enc.eval()
    if clf is not None:
        clf.eval()

    engine = InferenceEngine(
        InferenceConfig(eta=eta_final, delta=delta_final, topk=cfg.topk, device=device_str),
        hierarchy_obj
    )

    mem = None
    if cfg.use_memory:
        Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr_full, cfg, hd)
        mem = SemanticMemory(mem_cfg_final)
        mem.build(X_mem_base, Z_eval, Y_mem_base)


    X_te_enc = encode_with_encoder(enc, test_tokens, cfg.batch_size, device)
    X_te_dev = X_te_enc.to(device)
    with torch.no_grad():
        if clf is not None:
            p_cls_te = clf(X_te_dev)["p_cls"]
        else:
            p_cls_te = torch.zeros(X_te_dev.size(0), L, device=device)
        s_mem_te = mem.batch_query(X_te_dev, top_b=top_b_final) if mem is not None else torch.zeros_like(p_cls_te)
    pred_te = predict_with_strategy(
        s_mem=s_mem_te,
        p_cls=p_cls_te,
        engine=engine,
        cfg=cfg,
        eta_override=eta_final,
        delta_override=delta_final,
    )

    y_true_te = (Y_te.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_te = pred_te.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_te, y_pred_te)
    macro_all = macro_f1(y_true_te, y_pred_te)
    print("[TEST] Per-label metrics:")
    per_label_report(y_true_te, y_pred_te, hd.id2label)
    print(f"[Final Tuning] Using eta={eta_final:.2f}, delta={delta_final:.2f}, rho={rho_final:.2f}, top_b={top_b_final} derived from best fold. "
          f"(rho=標籤/樣本參數, eta=記憶/分類融合參數, delta=二值化閾值)")
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")

    # Save artifacts for reproducibility
    if mem is not None:
        mem.save(os.path.join(cfg.workdir, "memory_store"))
    # Save full-train checkpoint for downstream use (encoder + classifier)
    full_ckpt_path = os.path.join(cfg.workdir, "best_model_full.pt")
    torch.save({
        "encoder_state": enc.state_dict(),
        "classifier_state": (clf.state_dict() if clf is not None else None),
        "clf_cfg": clf_cfg_full.__dict__,
        "eta": eta_final,
        "delta": delta_final,
        "top_b": top_b_final,
        "memory_only": (clf is None) and bool(cfg.use_memory),
    }, full_ckpt_path)
    print(f"[Save] Full-train checkpoint saved to {full_ckpt_path}")
    with open(os.path.join(cfg.workdir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": hd.label2id, "id2label": {int(k): v for k, v in hd.id2label.items()}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.workdir, "ancestors.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in hd.ancestors.items()}, f, ensure_ascii=False, indent=2)
    print(f"[Done] Artifacts saved to {cfg.workdir}")

    scenario = scenario_name if scenario_name is not None else os.path.basename(cfg.workdir.rstrip(os.sep))
    return {
        "scenario": scenario,
        "eta": eta_final,
        "delta": delta_final,
        "rho": (rho_final if cfg.use_memory else None),
        "top_b": (top_b_final if cfg.use_memory else None),
        "micro": micro,
        "macro_all": macro_all,
        "use_memory": bool(cfg.use_memory),
        "use_global_branch": bool(cfg.use_global_branch),
        "use_local_branch": bool(cfg.use_local_branch),
    }


def train_single_fold(
    fold_name: str,
    cfg: TrainConfig,
    hd,
    hierarchy_obj: Hierarchy,
    level_slices,
    label_levels,
    same_level_map,
    edges_pc,
    label_tokens: Dict[str, torch.Tensor],
    train_tokens_full: Dict[str, torch.Tensor],
    Y_train_full: torch.Tensor,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    device_str: str,
    mem_cfg: MemoryConfig
) -> Dict[str, object]:
    print(f"\n[Fold {fold_name}] Training on {len(train_indices)} samples, validating on {len(val_indices)} samples")
    cls_on = classifier_enabled(cfg)
    enc = SharedEncoder(EncoderConfig(model_name=cfg.model_name, max_length=cfg.max_len, pooling="cls", normalize=True, device=device_str))
    clf_cfg = ClassifierConfig(
        hidden_size=enc.hidden_size,
        level_sizes=hd.level_sizes,
        level_slices=level_slices,
        dropout=cfg.dropout,
        global_head_mode=cfg.global_head_mode,
        global_hidden_ratio=cfg.global_hidden_ratio,
        use_global_branch=cfg.use_global_branch,
        use_local_branch=cfg.use_local_branch,
        device=device_str,
        loss=LossConfig(
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            use_bce_loss=True,
            path_on_local=False,
            tau_align=cfg.tau_align,
            tau_label=cfg.tau_label,
            num_neg_align=cfg.num_neg_align,
            num_neg_label=cfg.num_neg_label,
            use_align_loss=cfg.use_align_loss,
            use_label_loss=cfg.use_label_loss,
            use_path_loss=cfg.use_path_loss,
            weight_label=cfg.weight_label,
            weight_align=cfg.weight_align,
            weight_path=cfg.weight_path,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    label_tokens_device = move_tokens_to_device(label_tokens, device)
    comb = JointLossCombiner(clf_cfg).to(device)
    engine = InferenceEngine(
        InferenceConfig(eta=cfg.eta, delta=cfg.delta, topk=cfg.topk, device=device_str),
        hierarchy_obj
    )

    train_tokens = subset_tokens(train_tokens_full, train_indices)
    val_tokens = subset_tokens(train_tokens_full, val_indices)
    Y_tr = Y_train_full.index_select(0, torch.tensor(train_indices, dtype=torch.long))
    Y_va = Y_train_full.index_select(0, torch.tensor(val_indices, dtype=torch.long))

    tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100, level_mask = build_tail_level_masks(
        Y_tr, cfg.level_threshold, label_levels
    )
    sample_weights = compute_sample_weights(
        Y_tr,
        tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100, level_mask,
        cfg.tail_weight_q0_25, cfg.tail_weight_q25_50, cfg.tail_weight_q50_75, cfg.tail_weight_q75_100,
        cfg.level_weight
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count_stage1 = max(0, int(cfg.weighted_extra_ratio_stage1 * Ntr))
    extra_steps_stage1 = math.ceil(extra_sample_count_stage1 / B) if extra_sample_count_stage1 > 0 else 0
    steps_per_epoch_stage1 = base_steps + extra_steps_stage1
    extra_sample_count_stage2 = max(0, int(cfg.weighted_extra_ratio_stage2 * Ntr))
    extra_steps_stage2 = math.ceil(extra_sample_count_stage2 / B) if extra_sample_count_stage2 > 0 else 0
    steps_per_epoch_stage2 = base_steps + extra_steps_stage2

    pos_tr = to_pos_idx_list(Y_tr, label_levels)
    num_labels = int(sum(hd.level_sizes))
    val_metric = get_val_metric_name(cfg)
    best_score = -1.0
    best_val_micro = -1.0
    best_val_macro = -1.0
    best_loss = float("inf")
    best_path = os.path.join(cfg.workdir, f"best_model_{fold_name}.pt")
    patience = 3
    stale = 0
    last_tuned_eta = cfg.eta
    last_tuned_delta = cfg.delta
    last_tuned_rho = cfg.rho
    last_tuned_top_b = cfg.top_b

    def run_indices(batch_indices: List[int], opt, scheduler, running: Dict[str, float], trainable_params: List[torch.nn.Parameter]):
        need_cls = (clf is not None) and (
            getattr(comb.loss, "use_bce_loss", True)
            or (getattr(comb.loss, "use_path_loss", False) and getattr(comb.loss, "weight_path", 0.0) != 0.0)
        )
        need_z = (
            (getattr(comb.loss, "use_align_loss", False) and getattr(comb.loss, "weight_align", 0.0) != 0.0)
            or (
                getattr(comb.loss, "use_label_loss", False)
                and getattr(comb.loss, "weight_label", 0.0) != 0.0
                and getattr(comb, "label_loss_fn", None) is not None
            )
        )
        for start in range(0, len(batch_indices), B):
            batch_idx = batch_indices[start:start + B]
            if not batch_idx:
                continue
            batch_tokens = select_tokens_by_index(train_tokens, batch_idx, device)
            batch_kwargs = {
                "input_ids": batch_tokens["input_ids"],
                "attention_mask": batch_tokens["attention_mask"],
            }
            if "token_type_ids" in batch_tokens:
                batch_kwargs["token_type_ids"] = batch_tokens["token_type_ids"]
            h_x = enc.forward(**batch_kwargs)

            Z = None
            if need_z:
                label_kwargs = {
                    "input_ids": label_tokens_device["input_ids"],
                    "attention_mask": label_tokens_device["attention_mask"],
                }
                if "token_type_ids" in label_tokens_device:
                    label_kwargs["token_type_ids"] = label_tokens_device["token_type_ids"]
                Z = enc.forward(**label_kwargs)

            batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long)
            Yb = Y_tr.index_select(0, batch_idx_tensor).to(device)
            pos_b = [pos_tr[i] for i in batch_idx]

            if need_cls:
                out = clf(h_x)
                p_cls = out["p_cls"]
                p_local = out.get("p_local")
                logits_sum = out.get("logits_sum")
            else:
                p_cls = torch.zeros(h_x.size(0), num_labels, device=device)
                p_local = None
                logits_sum = torch.zeros_like(p_cls)

            losses = comb(
                p_cls=p_cls,
                logits_cls=logits_sum,
                p_local=p_local,
                Y=Yb,
                mask=None,
                h_x=h_x,
                Z=Z,
                pos_indices_per_sample=pos_b,
                edges_parent_child=edges_pc,
                Z_for_label_loss=Z,
                same_level_map=same_level_map,
                label_levels=label_levels,
            )
            loss = losses["loss_total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            scheduler.step()

            bs = len(batch_idx)
            running["bce"] += float(losses["loss_bce"]) * bs
            running["align"] += float(losses["loss_align"]) * bs
            running["path"] += float(losses["loss_path"]) * bs
            running["label"] += float(losses["loss_label"]) * bs
            running["total"] += float(loss) * bs

    # ---------------- Stage 1: Contrastive/align pretrain ----------------
    if cfg.contrast_epochs > 0:
        comb.loss.use_bce_loss = False
        comb.loss.use_align_loss = bool(cfg.use_align_loss)
        comb.loss.use_label_loss = bool(cfg.use_label_loss)
        comb.loss.use_path_loss = False
        comb.loss.weight_align = cfg.weight_align if cfg.use_align_loss else 0.0
        comb.loss.weight_label = cfg.weight_label if cfg.use_label_loss else 0.0
        comb.loss.weight_path = 0.0

        total_steps_contrast = max(1, steps_per_epoch_stage1 * cfg.contrast_epochs)
        warmup_steps_contrast = int(cfg.warmup_ratio * total_steps_contrast)
        opt_contrast = torch.optim.AdamW(enc.parameters(), lr=cfg.contrast_lr)
        scheduler_contrast = get_linear_schedule_with_warmup(
            opt_contrast,
            num_warmup_steps=warmup_steps_contrast,
            num_training_steps=total_steps_contrast
        )
        trainable_params_contrast = list(enc.parameters())

        for ep in range(1, cfg.contrast_epochs + 1):
            enc.train()
            if clf is not None:
                clf.train()
            running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "total": 0.0}

            base_indices = torch.randperm(Ntr).tolist()
            run_indices(base_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            if extra_sample_count_stage1 > 0:
                extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage1, replacement=True).tolist()
                run_indices(extra_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            print(f"[{fold_name} | Contrast Ep {ep}] train total={running['total']/Ntr:.4f} | "
                  f"align={running['align']/Ntr:.4f} label={running['label']/Ntr:.4f}")

            # For memory-only (no M3), validate after each contrast epoch and save best encoder.
            if (not cls_on) and cfg.use_memory:
                enc.eval()
                tuned_rho = cfg.rho
                tuned_delta = cfg.delta
                tuned_top_b = cfg.top_b
                best_score_epoch = -1.0
                best_s_mem_va: Optional[torch.Tensor] = None

                Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
                X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
                X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr, cfg, hd)
                X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
                X_va_dev = X_va_enc.to(device)

                rho_candidates = list(getattr(cfg, "rho_candidates", None) or [cfg.rho])
                if cfg.rho not in rho_candidates:
                    rho_candidates.append(cfg.rho)

                top_b_candidates = get_top_b_candidates(cfg)
                for rho_val in rho_candidates:
                    mem_tmp = SemanticMemory(mem_cfg)
                    mem_tmp.build(X_mem_base, Z_eval, Y_mem_base, rho=rho_val)
                    for top_b in top_b_candidates:
                        with torch.no_grad():
                            s_mem_va = mem_tmp.batch_query(X_va_dev, top_b=top_b)
                        delta_val, score, _, _ = tune_memory_only_delta(s_mem_va, Y_va, cfg)
                        if score > best_score_epoch:
                            best_score_epoch = score
                            tuned_rho = float(rho_val)
                            tuned_delta = float(delta_val)
                            tuned_top_b = int(top_b)
                            best_s_mem_va = s_mem_va.detach()

                y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
                L = int(sum(hd.level_sizes))
                s_for_eval = best_s_mem_va if best_s_mem_va is not None else torch.zeros(X_va_dev.size(0), L, device=X_va_dev.device)
                y_pred_va = (s_for_eval.detach().cpu().numpy() >= tuned_delta).astype(np.int32)
                micro = micro_f1(y_true_va, y_pred_va)
                macro_all = macro_f1(y_true_va, y_pred_va)
                print(f"[{fold_name} | Contrast Ep {ep}] VAL  micro-F1={micro:.4f}  macro-F1={macro_all:.4f}  "
                      f"(memory_only: rho={tuned_rho:.2f}, delta={tuned_delta:.2f}, top_b={tuned_top_b})")

                score = macro_all if val_metric == "macro" else micro
                if score > best_score:
                    best_score = score
                    best_val_micro = micro
                    best_val_macro = macro_all
                    last_tuned_eta = 1.0
                    last_tuned_delta = tuned_delta
                    last_tuned_rho = tuned_rho
                    last_tuned_top_b = tuned_top_b
                    torch.save({
                        "encoder_state": enc.state_dict(),
                        "classifier_state": None,
                        "clf_cfg": clf_cfg.__dict__,
                        "val_micro_f1": micro,
                        "epoch": ep,
                        "memory_only": True,
                        "rho": tuned_rho,
                        "eta": 1.0,
                        "delta": tuned_delta,
                        "top_b": tuned_top_b,
                    }, best_path)
                    print(f"  -> saved best checkpoint to {best_path}")
                    stale = 0
                else:
                    stale += 1

                if stale >= patience:
                    print(f"[{fold_name}] Early stopping triggered after {patience} stale epochs.")
                    break

    # ---------------- Stage 2: Classifier training ----------------
    if not cls_on:
        # If no contrast epochs (or never improved), do a single memory-only validation pass.
        if cfg.use_memory and (best_score < 0.0) and (not os.path.exists(best_path)):
            enc.eval()
            Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
            X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
            X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr, cfg, hd)
            X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
            X_va_dev = X_va_enc.to(device)

            tuned_rho = cfg.rho
            tuned_delta = cfg.delta
            tuned_top_b = cfg.top_b
            best_score_epoch = -1.0
            best_s_mem_va: Optional[torch.Tensor] = None

            rho_candidates = list(getattr(cfg, "rho_candidates", None) or [cfg.rho])
            if cfg.rho not in rho_candidates:
                rho_candidates.append(cfg.rho)

            top_b_candidates = get_top_b_candidates(cfg)
            for rho_val in rho_candidates:
                mem_tmp = SemanticMemory(mem_cfg)
                mem_tmp.build(X_mem_base, Z_eval, Y_mem_base, rho=rho_val)
                for top_b in top_b_candidates:
                    with torch.no_grad():
                        s_mem_va = mem_tmp.batch_query(X_va_dev, top_b=top_b)
                    delta_val, score, _, _ = tune_memory_only_delta(s_mem_va, Y_va, cfg)
                    if score > best_score_epoch:
                        best_score_epoch = score
                        tuned_rho = float(rho_val)
                        tuned_delta = float(delta_val)
                        tuned_top_b = int(top_b)
                        best_s_mem_va = s_mem_va.detach()

            last_tuned_eta = 1.0
            last_tuned_delta = tuned_delta
            last_tuned_rho = tuned_rho
            last_tuned_top_b = tuned_top_b

            y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
            L = int(sum(hd.level_sizes))
            s_for_eval = best_s_mem_va if best_s_mem_va is not None else torch.zeros(X_va_dev.size(0), L, device=X_va_dev.device)
            y_pred_va = (s_for_eval.detach().cpu().numpy() >= tuned_delta).astype(np.int32)
            micro = micro_f1(y_true_va, y_pred_va)
            macro_all = macro_f1(y_true_va, y_pred_va)
            best_score = macro_all if val_metric == "macro" else micro
            best_val_micro = micro
            best_val_macro = macro_all
            torch.save({
                "encoder_state": enc.state_dict(),
                "classifier_state": None,
                "clf_cfg": clf_cfg.__dict__,
                "val_micro_f1": micro,
                "epoch": 0,
                "memory_only": True,
                "rho": tuned_rho,
                "eta": 1.0,
                "delta": tuned_delta,
                "top_b": tuned_top_b,
            }, best_path)
            print(f"[{fold_name}] Saved memory-only checkpoint to {best_path} (micro-F1={micro:.4f}, rho={tuned_rho:.2f}, delta={tuned_delta:.2f}, top_b={tuned_top_b})")

        return {
            "fold": fold_name,
            "best_val_score": best_score,
            "best_val_micro": best_val_micro,
            "best_val_macro": best_val_macro,
            "best_path": best_path if os.path.exists(best_path) else None,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "last_tuned_eta": last_tuned_eta,
            "last_tuned_delta": last_tuned_delta,
            "last_tuned_rho": last_tuned_rho,
            "last_tuned_top_b": last_tuned_top_b,
            "use_memory": bool(cfg.use_memory),
        }

    comb.loss.use_bce_loss = True  # BCE/Focal 強制開
    comb.loss.use_align_loss = False
    comb.loss.use_label_loss = False
    comb.loss.use_path_loss = cfg.use_path_loss
    comb.loss.weight_align = 0.0
    comb.loss.weight_label = 0.0
    comb.loss.weight_path = cfg.weight_path if cfg.use_path_loss else 0.0

    cls_only_delta = 0.5
    def eval_classifier_only(ep_label: str) -> Tuple[float, float]:
        enc.eval()
        if clf is not None:
            clf.eval()
        X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
        X_va_dev = X_va_enc.to(device)
        with torch.no_grad():
            p_cls_va = clf(X_va_dev)["p_cls"] if clf is not None else torch.zeros(
                X_va_dev.size(0), int(sum(hd.level_sizes)), device=device
            )
        y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
        y_pred_va = (p_cls_va.detach().cpu().numpy() >= cls_only_delta).astype(np.int32)
        micro = micro_f1(y_true_va, y_pred_va)
        macro_all = macro_f1(y_true_va, y_pred_va)
        print(f"[{fold_name} | Cls-Only Ep {ep_label}] micro-F1={micro:.4f}  macro-F1={macro_all:.4f}  "
              f"(delta={cls_only_delta:.2f})")
        return micro, macro_all

    param_groups = [
        {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr matches stage 1
        {"params": clf.parameters(), "lr": cfg.classifier_lr},
    ]
    trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
    total_steps_cls = max(1, steps_per_epoch_stage2 * cfg.classifier_epochs)
    warmup_steps_cls = int(cfg.warmup_ratio * total_steps_cls)
    opt = torch.optim.AdamW(param_groups, lr=cfg.classifier_lr)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps_cls,
        num_training_steps=total_steps_cls
    )

    last_cls_eval_ep = 0
    final_epoch = 0
    for ep in range(1, cfg.classifier_epochs + 1):
        enc.train()
        if clf is not None:
            clf.train()
        running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_indices(base_indices, opt, scheduler, running, trainable_params_cls)

        if extra_sample_count_stage2 > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage2, replacement=True).tolist()
            run_indices(extra_indices, opt, scheduler, running, trainable_params_cls)

        print(f"[{fold_name} | Cls Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} path={running['path']/Ntr:.4f}")

        if ep % 4 == 0:
            eval_classifier_only(str(ep))
            last_cls_eval_ep = ep
        final_epoch = ep

        avg_loss = running["total"] / Ntr
        stale = 0 if avg_loss < best_loss - 1e-4 else stale + 1
        best_loss = min(best_loss, avg_loss)
        if stale >= patience:
            print(f"[{fold_name}] Early stopping triggered after {patience} stale epochs.")
            break

    if final_epoch > 0 and last_cls_eval_ep != final_epoch:
        eval_classifier_only(str(final_epoch))

    # After classifier training, tune memory fusion once on val set
    enc.eval()
    if clf is not None:
        clf.eval()

    mem = None
    tuned_eta = cfg.eta
    tuned_delta = cfg.delta
    tuned_rho = cfg.rho
    tuned_top_b = cfg.top_b
    tuned_score = -1.0
    if cfg.use_memory:
        Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr, cfg, hd)
        rho_candidates = getattr(cfg, "rho_candidates", None) or [cfg.rho]
        rho_candidates = list(rho_candidates)
        if cfg.rho not in rho_candidates:
            rho_candidates.append(cfg.rho)
        fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)

        best_mem = None
        best_rho = None
        best_eta = tuned_eta
        best_delta = tuned_delta
        best_top_b = tuned_top_b
        best_score = -1.0

        for rho_val in rho_candidates:
            mem_tmp = SemanticMemory(mem_cfg)
            mem_tmp.build(X_mem_base, Z_eval, Y_mem_base, rho=rho_val)

            if fusion_on:
                eta_val, delta_val, score, micro, macro, top_b_val = tune_fusion_parameters(
                    enc, clf, mem_tmp, engine, val_tokens, Y_va, cfg, device
                )
            else:
                eta_val, delta_val, score, micro, macro, top_b_val = tuned_eta, tuned_delta, -1.0, -1.0, -1.0, cfg.top_b

            if best_mem is None or score > best_score:
                best_score = score
                best_mem = mem_tmp
                best_rho = rho_val
                best_eta = eta_val
                best_delta = delta_val
                best_top_b = top_b_val

        mem = best_mem
        tuned_rho = best_rho if best_rho is not None else cfg.rho
        tuned_eta = best_eta
        tuned_delta = best_delta
        tuned_top_b = best_top_b
        tuned_score = best_score

    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if fusion_on and mem is not None:
        last_tuned_eta = tuned_eta
        last_tuned_delta = tuned_delta
        last_tuned_rho = tuned_rho
        last_tuned_top_b = tuned_top_b
        engine.cfg.eta = tuned_eta
        engine.cfg.delta = tuned_delta

    X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"] if clf is not None else torch.zeros(X_va_dev.size(0), int(sum(hd.level_sizes)), device=device)
        s_mem_va = mem.batch_query(X_va_dev, top_b=tuned_top_b) if mem is not None else torch.zeros_like(p_cls_va)
    if not cfg.use_memory:
        tuned_delta, tuned_score, _, _ = tune_classifier_only_delta(p_cls_va, Y_va, cfg)
        last_tuned_delta = tuned_delta
    pred_va = predict_with_strategy(
        s_mem=s_mem_va,
        p_cls=p_cls_va,
        engine=engine,
        cfg=cfg,
        eta_override=tuned_eta,
        delta_override=tuned_delta,
    )
    y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_va = pred_va.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_va, y_pred_va)
    macro_all = macro_f1(y_true_va, y_pred_va)
    best_score = macro_all if val_metric == "macro" else micro
    best_val_micro = micro
    best_val_macro = macro_all
    fusion_info = (f"(rho={tuned_rho:.2f}, eta={tuned_eta:.2f}, delta={tuned_delta:.2f}, top_b={tuned_top_b})"
                   if (fusion_on and mem is not None) else "(fusion off)")
    print(f"[{fold_name}] VAL (post-train tuning) micro-F1={micro:.4f}  macro-F1={macro_all:.4f}  {fusion_info}")

    torch.save({
        "encoder_state": enc.state_dict(),
        "classifier_state": clf.state_dict(),
        "clf_cfg": clf_cfg.__dict__,
        "val_micro_f1": micro,
        "epoch": cfg.classifier_epochs,
        "rho": tuned_rho,
        "eta": tuned_eta,
        "delta": tuned_delta,
        "top_b": tuned_top_b,
    }, best_path)
    print(f"  -> saved checkpoint to {best_path}")

    return {
        "fold": fold_name,
        "best_val_score": best_score,
        "best_val_micro": best_val_micro,
        "best_val_macro": best_val_macro,
        "best_path": best_path if os.path.exists(best_path) else None,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "last_tuned_eta": last_tuned_eta,
        "last_tuned_delta": last_tuned_delta,
        "last_tuned_rho": last_tuned_rho,
        "last_tuned_top_b": last_tuned_top_b,
        "use_memory": bool(cfg.use_memory),
    }


def train_full_model(
    cfg: TrainConfig,
    hd,
    hierarchy_obj: Hierarchy,
    level_slices,
    label_levels,
    same_level_map,
    edges_pc,
    label_tokens: Dict[str, torch.Tensor],
    train_tokens_full: Dict[str, torch.Tensor],
    Y_train_full: torch.Tensor,
    device: torch.device,
    device_str: str,
    mem_cfg: MemoryConfig
) -> Tuple[SharedEncoder, Optional[DualBranchHierClassifier], ClassifierConfig]:
    print("\n[Full Train] Training on all stratified training samples...")
    cls_on = classifier_enabled(cfg)
    enc = SharedEncoder(EncoderConfig(model_name=cfg.model_name, max_length=cfg.max_len, pooling="cls", normalize=True, device=device_str))
    clf_cfg = ClassifierConfig(
        hidden_size=enc.hidden_size,
        level_sizes=hd.level_sizes,
        level_slices=level_slices,
        dropout=cfg.dropout,
        global_head_mode=cfg.global_head_mode,
        global_hidden_ratio=cfg.global_hidden_ratio,
        use_global_branch=cfg.use_global_branch,
        use_local_branch=cfg.use_local_branch,
        device=device_str,
        loss=LossConfig(
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            use_bce_loss=True,
            path_on_local=False,
            tau_align=cfg.tau_align,
            tau_label=cfg.tau_label,
            num_neg_align=cfg.num_neg_align,
            num_neg_label=cfg.num_neg_label,
            use_align_loss=cfg.use_align_loss,
            use_label_loss=cfg.use_label_loss,
            use_path_loss=cfg.use_path_loss,
            weight_label=cfg.weight_label,
            weight_align=cfg.weight_align,
            weight_path=cfg.weight_path,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    comb = JointLossCombiner(clf_cfg).to(device)
    label_tokens_device = move_tokens_to_device(label_tokens, device)

    train_tokens = {k: v.to(device) for k, v in train_tokens_full.items()}
    Y_tr = Y_train_full.to(device)

    tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100, level_mask = build_tail_level_masks(
        Y_tr, cfg.level_threshold, label_levels
    )
    sample_weights = compute_sample_weights(
        Y_tr,
        tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100, level_mask,
        cfg.tail_weight_q0_25, cfg.tail_weight_q25_50, cfg.tail_weight_q50_75, cfg.tail_weight_q75_100,
        cfg.level_weight
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count_stage1 = max(0, int(cfg.weighted_extra_ratio_stage1 * Ntr))
    extra_steps_stage1 = math.ceil(extra_sample_count_stage1 / B) if extra_sample_count_stage1 > 0 else 0
    steps_per_epoch_stage1 = base_steps + extra_steps_stage1
    extra_sample_count_stage2 = max(0, int(cfg.weighted_extra_ratio_stage2 * Ntr))
    extra_steps_stage2 = math.ceil(extra_sample_count_stage2 / B) if extra_sample_count_stage2 > 0 else 0
    steps_per_epoch_stage2 = base_steps + extra_steps_stage2

    pos_tr = to_pos_idx_list(Y_tr, label_levels)
    num_labels = int(sum(hd.level_sizes))

    def run_indices(batch_indices: List[int], opt, scheduler, running: Dict[str, float], trainable_params: List[torch.nn.Parameter]):
        need_cls = (clf is not None) and (
            getattr(comb.loss, "use_bce_loss", True)
            or (getattr(comb.loss, "use_path_loss", False) and getattr(comb.loss, "weight_path", 0.0) != 0.0)
        )
        need_z = (
            (getattr(comb.loss, "use_align_loss", False) and getattr(comb.loss, "weight_align", 0.0) != 0.0)
            or (
                getattr(comb.loss, "use_label_loss", False)
                and getattr(comb.loss, "weight_label", 0.0) != 0.0
                and getattr(comb, "label_loss_fn", None) is not None
            )
        )
        for start in range(0, len(batch_indices), B):
            batch_idx = batch_indices[start:start + B]
            if not batch_idx:
                continue
            idx_tensor = torch.tensor(batch_idx, dtype=torch.long, device=device)
            batch_kwargs = {
                "input_ids": train_tokens["input_ids"].index_select(0, idx_tensor),
                "attention_mask": train_tokens["attention_mask"].index_select(0, idx_tensor),
            }
            if "token_type_ids" in train_tokens:
                batch_kwargs["token_type_ids"] = train_tokens["token_type_ids"].index_select(0, idx_tensor)
            h_x = enc.forward(**batch_kwargs)

            Z = None
            if need_z:
                label_kwargs = {
                    "input_ids": label_tokens_device["input_ids"],
                    "attention_mask": label_tokens_device["attention_mask"],
                }
                if "token_type_ids" in label_tokens_device:
                    label_kwargs["token_type_ids"] = label_tokens_device["token_type_ids"]
                Z = enc.forward(**label_kwargs)

            Yb = Y_tr.index_select(0, idx_tensor)
            pos_b = [pos_tr[i] for i in batch_idx]

            if need_cls:
                out = clf(h_x)
                p_cls = out["p_cls"]
                p_local = out.get("p_local")
                logits_sum = out.get("logits_sum")
            else:
                p_cls = torch.zeros(h_x.size(0), num_labels, device=device)
                p_local = None
                logits_sum = torch.zeros_like(p_cls)

            losses = comb(
                p_cls=p_cls,
                logits_cls=logits_sum,
                p_local=p_local,
                Y=Yb,
                mask=None,
                h_x=h_x,
                Z=Z,
                pos_indices_per_sample=pos_b,
                edges_parent_child=edges_pc,
                Z_for_label_loss=Z,
                same_level_map=same_level_map,
                label_levels=label_levels,
            )
            loss = losses["loss_total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            scheduler.step()

            bs = len(batch_idx)
            running["bce"] += float(losses["loss_bce"]) * bs
            running["align"] += float(losses["loss_align"]) * bs
            running["path"] += float(losses["loss_path"]) * bs
            running["label"] += float(losses["loss_label"]) * bs
            running["total"] += float(loss) * bs

    # ---------------- Stage 1: Contrastive/align pretrain ----------------
    if cfg.contrast_epochs > 0:
        comb.loss.use_bce_loss = False
        comb.loss.use_align_loss = bool(cfg.use_align_loss)
        comb.loss.use_label_loss = bool(cfg.use_label_loss)
        comb.loss.use_path_loss = False
        comb.loss.weight_align = cfg.weight_align if cfg.use_align_loss else 0.0
        comb.loss.weight_label = cfg.weight_label if cfg.use_label_loss else 0.0
        comb.loss.weight_path = 0.0

        total_steps_contrast = max(1, steps_per_epoch_stage1 * cfg.contrast_epochs)
        warmup_steps_contrast = int(cfg.warmup_ratio * total_steps_contrast)
        opt_contrast = torch.optim.AdamW(enc.parameters(), lr=cfg.contrast_lr)
        scheduler_contrast = get_linear_schedule_with_warmup(
            opt_contrast,
            num_warmup_steps=warmup_steps_contrast,
            num_training_steps=total_steps_contrast
        )
        trainable_params_contrast = list(enc.parameters())

        for ep in range(1, cfg.contrast_epochs + 1):
            enc.train()
            if clf is not None:
                clf.train()
            running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "total": 0.0}

            base_indices = torch.randperm(Ntr).tolist()
            run_indices(base_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            if extra_sample_count_stage1 > 0:
                extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage1, replacement=True).tolist()
                run_indices(extra_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            print(f"[Full Train | Contrast Ep {ep}] train total={running['total']/Ntr:.4f} | "
                  f"align={running['align']/Ntr:.4f} label={running['label']/Ntr:.4f}")

    # ---------------- Stage 2: Classifier training ----------------
    if not cls_on:
        print("[Full Train] M3 disabled (memory_only): skipping classifier training stage.")
        return enc, None, clf_cfg

    comb.loss.use_bce_loss = True  # BCE/Focal 強制開
    comb.loss.use_align_loss = False
    comb.loss.use_label_loss = False
    comb.loss.use_path_loss = cfg.use_path_loss
    comb.loss.weight_align = 0.0
    comb.loss.weight_label = 0.0
    comb.loss.weight_path = cfg.weight_path if cfg.use_path_loss else 0.0

    assert clf is not None
    trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
    param_groups = [
        {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr matches stage 1
        {"params": clf.parameters(), "lr": cfg.classifier_lr},
    ]
    total_steps_cls = max(1, steps_per_epoch_stage2 * cfg.classifier_epochs)
    warmup_steps_cls = int(cfg.warmup_ratio * total_steps_cls)
    opt = torch.optim.AdamW(param_groups, lr=cfg.classifier_lr)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps_cls,
        num_training_steps=total_steps_cls
    )

    for ep in range(1, cfg.classifier_epochs + 1):
        enc.train()
        clf.train()
        running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_indices(base_indices, opt, scheduler, running, trainable_params_cls)

        if extra_sample_count_stage2 > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage2, replacement=True).tolist()
            run_indices(extra_indices, opt, scheduler, running, trainable_params_cls)

        print(f"[Full Train | Cls Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} path={running['path']/Ntr:.4f}")

    return enc, clf, clf_cfg


if __name__ == "__main__":
    main(TrainConfig())
