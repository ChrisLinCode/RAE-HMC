# train_rae_hmc.py
# End-to-end training + validation + test for RAE-HMC
# Splits train/val/test from a single dataset.csv and prints research metrics to console.
# Modules required in same folder: build_hierarchy_utils.py, encoder.py, memory.py, classifier.py, inference.py

import os, json, random, math
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from build_hierarchy_utils import (
    load_hierarchy_from_file, build_multi_hot_Y, make_level_slices, parse_label_hierarchy
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
    exclude_root_label: bool = True
    root_label_name: Union[str, List[str]] = field(default_factory=lambda: ["Root", "食材"])

    # Split ratio for test set (train:test = 1 - test_ratio : test_ratio)
    test_ratio: float = 0.2
    seed: int = 42

    # Encoder
    model_name: str = "bert-base-chinese"
    max_len: int = 32
    # Label text depth for memory label embeddings:
    # 0=self only, 1=parent>self, 2=grandparent>parent>self, ...
    label_path_depth: int = 1
    batch_size: int = 16 
    cache_tokens_on_gpu: bool = True  # cache fold tokens on GPU to reduce CPU->GPU transfer (needs VRAM)
    use_bf16_amp: bool = True
    encoder_pooling: str = "mean"  # "cls" | "mean"

    # Training hyperparameters
    classifier_epochs: int = 12
    contrast_lr: float = 5e-5   
    classifier_lr: float = 7e-3
    classifier_lr_global: Optional[float] = None 
    classifier_lr_local: Optional[float] = None 
    local_lr_scale: float = 0.1 #local 學習率太強會崩掉
    classifier_lr_fusion: Optional[float] = None
    
    warmup_ratio: float = 0.15 

    # contrastive losses (shared master switch)
    use_cl_loss: bool = True
    cl_tau: float = 0.15
    sample_cl_weight: float = 0.0001
    hnm_cl_weight: float = 0.01
    hnm_cl_pos_topm: int = 5
    hnm_cl_neg_topk: int = 10
    hnm_cl_refresh_every_epochs: int = 1
    
    # Memory (M2)
    tau_mem: float = 0.07 # memory retrieval temperature
    rho: float = 0.5
    top_b: int = 45
    rho_candidates: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    top_b_candidates: List[int] = field(default_factory=lambda: [15, 30, 45, 60, 75, 90])

    # Classifier (M3)
    local_num_heads: int = 2 #多頭注意力頭數

    hidden_ratio: float = 1.0
    global_hidden_ratio: Optional[float] = None
    local_head_hidden_ratio: Optional[float] = None
    fusion_hidden_ratio: Optional[float] = None

    dropout: float = 0.15
    global_dropout: Optional[float] = None
    local_dropout: Optional[float] = None    

    # 預設參數 = HMCN論文配置
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # path loss
    use_path_loss: bool = True
    weight_path: float = 1.0 

    # Fusion (M4)
    topk: Optional[int] = 15
    eta: float = 0.5
    delta: float = 0.3
    delta_mode: str = "global"  # "global" | "level" (per-level thresholds)    
    eta_candidates: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    delta_candidates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

    # Sampling (tail-aware / level-aware)
    # Four-bin tail weighting by fixed label frequency quartiles (0-25/25-50/50-75/75-100)
    tail_weight_q0_25: float = 1.75
    tail_weight_q25_50: float = 1.5
    tail_weight_q50_75: float = 1.25
    tail_weight_q75_100: float = 1.0

    level_weight_scale: float = 0.05  # per-sample max label depth * scale

    weighted_extra_ratio: float = 1.0 #1.0
    val_metric: str = "macro"  # metric for selecting best validation parameters ("micro" or "macro")
    auto_tune_params: bool = True  # If True, tune rho/eta/delta/top_b on validation

    # Module switches / ablations
    use_memory: bool = True
    use_local_branch: bool = True
    use_global_branch: bool = True
    run_ablation: str = "off"      # "off" | "fixed" | "best" (bool False/True maps to off/best)

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

def parse_root_label_names(root_label_name: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    if root_label_name is None:
        return []

    names: List[str] = []
    if isinstance(root_label_name, str):
        names = [t.strip() for t in root_label_name.replace(";", ",").split(",") if t.strip()]
    elif isinstance(root_label_name, (list, tuple)):
        for item in root_label_name:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            names.extend([t.strip() for t in text.replace(";", ",").split(",") if t.strip()])
    else:
        text = str(root_label_name).strip()
        if text:
            names = [text]

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped

def _root_payload_to_dict(payload) -> Dict[str, object]:
    if isinstance(payload, dict):
        return dict(payload)
    promoted: Dict[str, object] = {}
    if isinstance(payload, list):
        for elem in payload:
            if isinstance(elem, dict):
                for k, v in elem.items():
                    promoted[k] = v
            elif isinstance(elem, str):
                promoted[elem] = None
    elif isinstance(payload, str):
        promoted[payload] = None
    return promoted

def promote_named_roots(hjson: Dict[str, object], root_names: List[str]) -> Dict[str, object]:
    if not root_names:
        return hjson

    root_set = set(root_names)
    current = dict(hjson)

    # Repeatedly peel named roots from current top-level keys:
    # e.g., Root -> 食材 -> ... with root_names=["Root", "食材"].
    while any(k in root_set for k in current.keys()):
        promoted: Dict[str, object] = {}
        for k, v in current.items():
            if k in root_set:
                payload_dict = _root_payload_to_dict(v)
                for child_k, child_v in payload_dict.items():
                    if child_k not in promoted:
                        promoted[child_k] = child_v
            else:
                if k not in promoted:
                    promoted[k] = v

        # Safety guard: avoid infinite loop for pathological self-nested roots.
        if promoted == current:
            break
        current = promoted

    return current

def strip_root_label(labels: List[str], root_names: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    roots = parse_root_label_names(root_names)
    if not roots:
        return labels
    root_set = set(roots)
    return [l for l in labels if l not in root_set]

def build_label_descriptions(hd, path_depth: int) -> List[str]:
    """
    Build label text descriptions with controllable ancestor depth.
    depth=0 -> self
    depth=1 -> parent > self
    depth=2 -> grandparent > parent > self
    For multi-parent DAGs, follow the first parent (same convention as path_strings).
    """
    depth = max(0, int(path_depth))
    label_descs: List[str] = []
    for label_id in range(hd.num_labels):
        chain = [int(label_id)]
        cur = int(label_id)
        for _ in range(depth):
            parents = hd.parents.get(cur, [])
            if not parents:
                break
            cur = int(parents[0])
            chain.append(cur)
        chain.reverse()
        parts = [hd.id2label.get(i, str(i)) for i in chain]
        label_descs.append(" > ".join(parts))
    return label_descs

def init_ablation_result_txt(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Ablation Summary]\n")

def append_ablation_result_txt(path: str, item: Dict[str, float]) -> None:
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
    micro = item.get("micro", 0.0)
    macro = item.get("macro_all", 0.0)
    scenario = item.get("scenario", "")

    line = (
        f"  - {scenario}: eta={eta_print}, delta={delta_print}, rho={rho_print}, "
        f"top_b={top_b_print}, micro-F1={micro:.4f}, macro-F1(all)={macro:.4f}"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

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

def bf16_amp_enabled(cfg, device: torch.device) -> bool:
    return bool(getattr(cfg, "use_bf16_amp", False)) and device.type == "cuda"

def bf16_autocast_context(cfg, device: torch.device):
    if not bf16_amp_enabled(cfg, device):
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("use_bf16_amp=True but this CUDA device does not support bfloat16 autocast.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

def build_encoder_config(cfg: TrainConfig, device_str: str) -> EncoderConfig:
    return EncoderConfig(
        model_name=cfg.model_name,
        max_length=cfg.max_len,
        pooling=cfg.encoder_pooling,
        normalize=True,
        device=device_str,
        amp_enabled=bool(getattr(cfg, "use_bf16_amp", False)),
        amp_dtype="bf16",
        grad_checkpointing=bool(getattr(cfg, "grad_checkpointing", False)),
    )

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

def normalize_ablation_mode(value: object) -> str:
    if isinstance(value, bool):
        return "best" if value else "off"
    if value is None:
        return "off"
    text = str(value).strip().lower()
    if text in {"off", "fixed", "best"}:
        return text
    if text in {"true", "on", "yes", "1"}:
        return "best"
    if text in {"false", "no", "0"}:
        return "off"
    return "off"

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

def normalize_delta_mode(cfg: TrainConfig) -> str:
    mode = str(getattr(cfg, "delta_mode", "global")).lower().strip()
    return "level" if mode in {"level", "per_level", "levelwise"} else "global"

def build_delta_vector(
    label_levels: List[int],
    delta_level_map: Dict[int, float],
    default_delta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    deltas = [float(delta_level_map.get(int(lvl), default_delta)) for lvl in label_levels]
    return torch.tensor(deltas, device=device, dtype=dtype)

def build_topk_mask(scores: torch.Tensor, topk: Optional[int]) -> Optional[torch.Tensor]:
    if topk is None:
        return None
    k = max(1, int(topk))
    _, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(1, idx, True)
    return keep

def apply_delta(
    scores: torch.Tensor,
    delta: object,
    topk: Optional[int] = None
) -> torch.Tensor:
    if torch.is_tensor(delta):
        delta_t = delta.to(device=scores.device, dtype=scores.dtype)
        if delta_t.dim() == 1:
            delta_t = delta_t.view(1, -1)
    else:
        delta_t = float(delta)
    keep = build_topk_mask(scores, topk)
    if keep is not None:
        kept_scores = scores * keep
        return (kept_scores >= delta_t).to(torch.int64)
    return (scores >= delta_t).to(torch.int64)

def binarize_scores(
    scores: torch.Tensor,
    cfg: TrainConfig,
    label_levels: Optional[List[int]] = None,
    delta_override: Optional[float] = None,
    delta_level_map: Optional[Dict[int, float]] = None,
    topk_override: Optional[int] = None
) -> torch.Tensor:
    topk = topk_override
    mode = normalize_delta_mode(cfg)
    if mode == "level" and label_levels is not None and delta_level_map is not None:
        default_delta = float(cfg.delta if delta_override is None else delta_override)
        delta_vec = build_delta_vector(
            label_levels, delta_level_map, default_delta, scores.device, scores.dtype
        )
        return apply_delta(scores, delta_vec, topk=topk)
    delta_val = float(cfg.delta if delta_override is None else delta_override)
    return apply_delta(scores, delta_val, topk=topk)

def tune_level_delta_map(
    scores_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    label_levels: List[int],
    topk: Optional[int] = None,
) -> Tuple[Dict[int, float], float, float, float]:
    y_true = (Y_val.detach().cpu().numpy() > 0.5).astype(np.int32)
    candidates = list(getattr(cfg, "delta_candidates", None) or [cfg.delta])
    if cfg.delta not in candidates:
        candidates.append(cfg.delta)
    candidates = [float(max(0.0, min(1.0, t))) for t in candidates]

    scores_cpu = scores_val.detach().cpu()
    keep = build_topk_mask(scores_cpu, topk)

    level_to_cols: Dict[int, List[int]] = {}
    for idx, lvl in enumerate(label_levels):
        level_to_cols.setdefault(int(lvl), []).append(idx)

    delta_level_map: Dict[int, float] = {}
    for lvl, cols in level_to_cols.items():
        if not cols:
            continue
        y_true_lvl = y_true[:, cols]
        scores_lvl = scores_cpu[:, cols]
        if keep is not None:
            scores_lvl = scores_lvl * keep[:, cols]
        best_delta = float(cfg.delta)
        best_score = -1.0
        for delta_val in candidates:
            y_pred_lvl = (scores_lvl >= delta_val).numpy().astype(np.int32)
            score, _, _ = compute_val_metrics(y_true_lvl, y_pred_lvl, cfg)
            if score > best_score:
                best_score = score
                best_delta = float(delta_val)
        delta_level_map[int(lvl)] = best_delta

    delta_vec = build_delta_vector(label_levels, delta_level_map, cfg.delta, scores_cpu.device, scores_cpu.dtype)
    y_pred_all = apply_delta(scores_cpu, delta_vec, topk=topk).numpy().astype(np.int32)
    score, micro, macro = compute_val_metrics(y_true, y_pred_all, cfg)
    return delta_level_map, score, micro, macro

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
    rows = []
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
        rows.append((support, j, name, p, r, f1))

    for support, j, name, p, r, f1 in sorted(rows, key=lambda x: (-x[0], x[1])):
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
    label_levels: Optional[List[int]] = None,
    delta_levels_override: Optional[Dict[int, float]] = None,
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
        s_fused = engine.fuse_scores(s_mem, p_cls, eta=eta)
        return binarize_scores(
            s_fused,
            cfg,
            label_levels=label_levels,
            delta_override=delta,
            delta_level_map=delta_levels_override,
            topk_override=cfg.topk,
        )
    if cfg.use_memory:
        return binarize_scores(
            s_mem,
            cfg,
            label_levels=label_levels,
            delta_override=delta,
            delta_level_map=delta_levels_override,
        )
    return binarize_scores(
        p_cls,
        cfg,
        label_levels=label_levels,
        delta_override=delta,
        delta_level_map=delta_levels_override,
    )

def classifier_enabled(cfg: TrainConfig) -> bool:
    return bool(cfg.use_global_branch or cfg.use_local_branch)

def build_classifier_param_groups(
    clf: DualBranchHierClassifier,
    cfg: TrainConfig
) -> List[Dict[str, object]]:
    lr_global = cfg.classifier_lr_global if cfg.classifier_lr_global is not None else cfg.classifier_lr
    lr_local_default = cfg.classifier_lr * float(getattr(cfg, "local_lr_scale", 0.1))
    lr_local = cfg.classifier_lr_local if cfg.classifier_lr_local is not None else lr_local_default
    lr_fusion = cfg.classifier_lr_fusion if cfg.classifier_lr_fusion is not None else lr_global

    global_params = []
    if hasattr(clf, "global_head"):
        global_params.extend(list(clf.global_head.parameters()))
    fusion_params = []
    if hasattr(clf, "fusion_mlp"):
        fusion_params.extend(list(clf.fusion_mlp.parameters()))

    local_params = []
    for name in ("local_first", "local_attn", "local_norm", "local_ff", "local_heads"):
        module = getattr(clf, name, None)
        if module is None:
            continue
        local_params.extend(list(module.parameters()))

    seen = {id(p) for p in global_params + fusion_params + local_params}
    remaining = [p for p in clf.parameters() if id(p) not in seen]

    groups = []
    if global_params:
        groups.append({"params": global_params, "lr": lr_global})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": lr_fusion})
    if local_params:
        groups.append({"params": local_params, "lr": lr_local})
    if remaining:
        groups.append({"params": remaining, "lr": lr_global})
    return groups

def tune_memory_only_delta(
    s_mem_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    label_levels: Optional[List[int]] = None,
    topk: Optional[int] = None,
) -> Tuple[float, Optional[Dict[int, float]], float, float, float]:
    """
    Tune a single global delta for memory-only predictions:
        y_hat = 1[s_mem >= delta]
    Returns (best_delta, best_delta_levels, best_score, best_micro_f1, best_macro_f1).
    """
    if normalize_delta_mode(cfg) == "level" and label_levels is not None:
        delta_map, score, micro, macro = tune_level_delta_map(
            s_mem_val, Y_val, cfg, label_levels, topk=topk
        )
        return float(cfg.delta), delta_map, score, micro, macro
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
        y_pred = apply_delta(s_cpu, delta_val, topk=topk).numpy().astype(np.int32)
        score, micro, macro = compute_val_metrics(y_true, y_pred, cfg)
        if score > best_score:
            best_score = score
            best_micro = micro
            best_macro = macro
            best_delta = float(delta_val)
    return best_delta, None, best_score, best_micro, best_macro

def tune_classifier_only_delta(
    p_cls_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    label_levels: Optional[List[int]] = None,
    topk: Optional[int] = None,
) -> Tuple[float, Optional[Dict[int, float]], float, float, float]:
    """
    Tune a single global delta for classifier-only predictions:
        y_hat = 1[p_cls >= delta]
    Returns (best_delta, best_delta_levels, best_score, best_micro_f1, best_macro_f1).
    """
    if normalize_delta_mode(cfg) == "level" and label_levels is not None:
        delta_map, score, micro, macro = tune_level_delta_map(
            p_cls_val, Y_val, cfg, label_levels, topk=topk
        )
        return float(cfg.delta), delta_map, score, micro, macro
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
        y_pred = apply_delta(p_cpu, delta_val, topk=topk).numpy().astype(np.int32)
        score, micro, macro = compute_val_metrics(y_true, y_pred, cfg)
        if score > best_score:
            best_score = score
            best_micro = micro
            best_macro = macro
            best_delta = float(delta_val)
    return best_delta, None, best_score, best_micro, best_macro

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

def configure_cl_for_stage2(
    comb: JointLossCombiner,
    cfg: TrainConfig,
    classifier_on: bool = True,
) -> bool:
    stage2_cl_on = bool(cfg.use_cl_loss)
    comb.loss.use_cl_loss = stage2_cl_on
    comb.loss.use_bce_loss = bool(classifier_on)
    sample_cl_w = float(getattr(cfg, "sample_cl_weight", 0.0))
    comb.loss.weight_cl = sample_cl_w if stage2_cl_on else 0.0
    if classifier_on:
        comb.loss.use_path_loss = cfg.use_path_loss
        comb.loss.weight_path = cfg.weight_path if cfg.use_path_loss else 0.0
    else:
        comb.loss.use_path_loss = False
        comb.loss.weight_path = 0.0
    return stage2_cl_on

@dataclass
class LabelLossState:
    enabled: bool
    label_tokens: Dict[str, torch.Tensor]
    terminal_pos_by_sample: List[List[int]]
    exclude_by_sample: List[List[int]]
    tau: float
    topm: int
    topk: int
    weight: float
    label_cache: Optional[torch.Tensor] = None

def build_descendant_index_map(children: Dict[int, List[int]], num_labels: int) -> Dict[int, List[int]]:
    descendants: Dict[int, List[int]] = {}
    for nid in range(num_labels):
        seen = set()
        stack = list(children.get(nid, []))
        while stack:
            cur = int(stack.pop())
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))
        descendants[nid] = sorted(seen)
    return descendants

def build_terminal_and_exclude_indices(
    Y: torch.Tensor,
    ancestors: Dict[int, List[int]],
    descendants: Dict[int, List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    Y_cpu = (Y.detach().cpu() > 0.5)
    N, L = Y_cpu.shape
    terminal_by_sample: List[List[int]] = []
    exclude_by_sample: List[List[int]] = []

    for i in range(N):
        pos = torch.nonzero(Y_cpu[i], as_tuple=False).view(-1).tolist()
        pos_set = set(int(p) for p in pos)
        terminals: List[int] = []
        for p in pos:
            p_int = int(p)
            desc = descendants.get(p_int, [])
            has_pos_desc = any(int(d) in pos_set for d in desc)
            if not has_pos_desc:
                terminals.append(p_int)
        if not terminals:
            terminals = [int(p) for p in pos]

        exclude = set()
        for t in terminals:
            exclude.add(int(t))
            exclude.update(int(a) for a in ancestors.get(int(t), []))
            exclude.update(int(d) for d in descendants.get(int(t), []))

        terminals = sorted([t for t in terminals if 0 <= int(t) < L])
        exclude_sorted = sorted([e for e in exclude if 0 <= int(e) < L])
        terminal_by_sample.append(terminals)
        exclude_by_sample.append(exclude_sorted)
    return terminal_by_sample, exclude_by_sample

def compute_label_infonce_loss(
    *,
    enc: SharedEncoder,
    h_x: torch.Tensor,
    batch_indices: List[int],
    label_state: LabelLossState,
    device: torch.device,
) -> torch.Tensor:
    if not label_state.enabled or label_state.label_cache is None:
        return h_x.sum() * 0.0

    B = h_x.size(0)
    if B == 0:
        return h_x.sum() * 0.0

    tau = max(1e-8, float(label_state.tau))
    topm = max(1, int(label_state.topm))
    topk = max(1, int(label_state.topk))

    x_norm = F.normalize(h_x, p=2, dim=-1)
    z_cache = label_state.label_cache
    if z_cache.device != device:
        z_cache = z_cache.to(device)
    z_cache_norm = F.normalize(z_cache, p=2, dim=-1)
    sim_cache = x_norm @ z_cache_norm.T  # [B, L]

    chosen: List[Tuple[int, int, List[int]]] = []
    unique_label_ids = set()

    for row, sample_idx in enumerate(batch_indices):
        if sample_idx < 0 or sample_idx >= len(label_state.terminal_pos_by_sample):
            continue
        pos_pool = label_state.terminal_pos_by_sample[sample_idx]
        if not pos_pool:
            continue

        pos_tensor = torch.tensor(pos_pool, dtype=torch.long, device=device)
        pos_sims = sim_cache[row].index_select(0, pos_tensor)
        m = min(topm, int(pos_sims.numel()))
        if m <= 0:
            continue
        hardest_m = torch.topk(pos_sims, k=m, largest=False).indices
        pick = int(torch.randint(0, m, (1,), device=device).item())
        pos_idx_in_pool = int(hardest_m[pick].item())
        pos_label_id = int(pos_pool[pos_idx_in_pool])

        row_scores = sim_cache[row].clone()
        exclude = label_state.exclude_by_sample[sample_idx] if sample_idx < len(label_state.exclude_by_sample) else []
        if exclude:
            ex_tensor = torch.tensor(exclude, dtype=torch.long, device=device)
            row_scores.index_fill_(0, ex_tensor, float("-inf"))

        finite_count = int(torch.isfinite(row_scores).sum().item())
        if finite_count <= 0:
            continue
        k = min(topk, finite_count)
        neg_label_ids = torch.topk(row_scores, k=k, largest=True).indices.tolist()
        neg_label_ids = [int(v) for v in neg_label_ids]
        if not neg_label_ids:
            continue

        chosen.append((row, pos_label_id, neg_label_ids))
        unique_label_ids.add(pos_label_id)
        unique_label_ids.update(neg_label_ids)

    if not chosen:
        return h_x.sum() * 0.0

    unique_ids = sorted(list(unique_label_ids))
    label_batch_tokens = select_tokens_by_index(label_state.label_tokens, unique_ids, device)
    label_kwargs = {
        "input_ids": label_batch_tokens["input_ids"],
        "attention_mask": label_batch_tokens["attention_mask"],
    }
    if "token_type_ids" in label_batch_tokens:
        label_kwargs["token_type_ids"] = label_batch_tokens["token_type_ids"]
    z_online = enc.forward(**label_kwargs)
    z_online = F.normalize(z_online, p=2, dim=-1)

    id_to_col = {int(lid): col for col, lid in enumerate(unique_ids)}
    loss_terms: List[torch.Tensor] = []
    for row, pos_label_id, neg_label_ids in chosen:
        pos_col = id_to_col.get(pos_label_id, None)
        if pos_col is None:
            continue
        pos_logit = torch.dot(x_norm[row], z_online[pos_col]) / tau
        neg_cols = [id_to_col[nid] for nid in neg_label_ids if nid in id_to_col]
        if not neg_cols:
            continue
        neg_tensor = z_online.index_select(0, torch.tensor(neg_cols, dtype=torch.long, device=device))
        neg_logits = neg_tensor @ x_norm[row] / tau
        logits = torch.cat([pos_logit.view(1), neg_logits], dim=0)
        loss_i = -(pos_logit - torch.logsumexp(logits, dim=0))
        loss_terms.append(loss_i)

    if not loss_terms:
        return h_x.sum() * 0.0
    return torch.stack(loss_terms).mean()

def init_label_loss_state(
    cfg: TrainConfig,
    hd,
    Y_train_subset: torch.Tensor,
    label_tokens: Dict[str, torch.Tensor],
) -> Optional[LabelLossState]:
    # Keep sample CL and HNM-CL under one master switch.
    if not bool(getattr(cfg, "use_cl_loss", False)):
        return None
    weight = float(getattr(cfg, "hnm_cl_weight", 0.0))
    if weight <= 0.0:
        return None

    num_labels = int(Y_train_subset.size(1))
    descendants = build_descendant_index_map(getattr(hd, "children", {}), num_labels)
    terminal_pos, exclude = build_terminal_and_exclude_indices(
        Y_train_subset,
        getattr(hd, "ancestors", {}),
        descendants,
    )
    return LabelLossState(
        enabled=True,
        label_tokens=label_tokens,
        terminal_pos_by_sample=terminal_pos,
        exclude_by_sample=exclude,
        tau=float(getattr(cfg, "cl_tau", 0.1)),
        topm=int(getattr(cfg, "hnm_cl_pos_topm", 5)),
        topk=int(getattr(cfg, "hnm_cl_neg_topk", 16)),
        weight=weight,
        label_cache=None,
    )

def maybe_refresh_label_cache(
    label_state: Optional[LabelLossState],
    enc: SharedEncoder,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> None:
    if label_state is None or not label_state.enabled:
        return
    refresh_every = max(1, int(getattr(cfg, "hnm_cl_refresh_every_epochs", 2)))
    if label_state.label_cache is not None and ((epoch - 1) % refresh_every) != 0:
        return
    z_cache = encode_with_encoder(enc, label_state.label_tokens, cfg.batch_size, device)
    label_state.label_cache = F.normalize(z_cache.to(device), p=2, dim=-1).detach()

def run_training_indices_common(
    *,
    cfg: TrainConfig,
    enc: SharedEncoder,
    clf: Optional[DualBranchHierClassifier],
    comb: JointLossCombiner,
    train_tokens: Dict[str, torch.Tensor],
    Y_tr: torch.Tensor,
    batch_indices: List[int],
    B: int,
    device: torch.device,
    num_labels: int,
    edges_pc: List[Tuple[int, int]],
    trainable_params: List[torch.nn.Parameter],
    opt,
    scheduler,
    running: Dict[str, float],
    label_state: Optional[LabelLossState] = None,
) -> None:
    need_cls = (clf is not None) and (
        getattr(comb.loss, "use_bce_loss", True)
        or (getattr(comb.loss, "use_path_loss", False) and getattr(comb.loss, "weight_path", 0.0) != 0.0)
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
        with bf16_autocast_context(cfg, device):
            h_x = enc.forward(**batch_kwargs)

            idx_tensor = torch.tensor(batch_idx, dtype=torch.long, device=device)
            Yb = Y_tr.index_select(0, idx_tensor)

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
                edges_parent_child=edges_pc,
            )
            loss_label = h_x.sum() * 0.0
            if label_state is not None and label_state.enabled:
                loss_label = compute_label_infonce_loss(
                    enc=enc,
                    h_x=h_x,
                    batch_indices=batch_idx,
                    label_state=label_state,
                    device=device,
                )
            loss = losses["loss_total"] + (
                float(label_state.weight) * loss_label if (label_state is not None and label_state.enabled) else 0.0
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        scheduler.step()

        bs = len(batch_idx)
        running["bce"] += float(losses["loss_bce"]) * bs
        running["sample_cl_loss"] += float(losses["loss_cl"]) * bs
        running["path"] += float(losses["loss_path"]) * bs
        running["hnm_cl_loss"] += float(loss_label.detach()) * bs
        running["total"] += float(loss.detach()) * bs

def build_tail_level_masks(
    Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100

def compute_sample_weights(
    Y: torch.Tensor,
    tail_mask_q0_25: torch.Tensor,
    tail_mask_q25_50: torch.Tensor,
    tail_mask_q50_75: torch.Tensor,
    tail_mask_q75_100: torch.Tensor,
    tail_weight_q0_25: float,
    tail_weight_q25_50: float,
    tail_weight_q50_75: float,
    tail_weight_q75_100: float,
    label_levels: List[int],
    level_weight_scale: float
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
    if level_weight_scale != 0.0:
        lvl_tensor = torch.tensor(label_levels, device=Y.device, dtype=torch.float32)
        y_mask = (Y > 0.5).to(torch.float32)
        max_levels = (y_mask * lvl_tensor).max(dim=1).values
        weights += level_weight_scale * max_levels
    return weights

def build_memory_prototypes(
    X: torch.Tensor,
    Y: torch.Tensor,
    hd,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_cpu = X.detach().cpu()
    Y_cpu = Y.detach().cpu()
    L = int(Y_cpu.size(1))
    ancestors = getattr(hd, "ancestors", {})

    X_out: List[torch.Tensor] = []
    Y_out: List[torch.Tensor] = []

    for label in range(L):
        idx = torch.nonzero(Y_cpu[:, label] > 0.5, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        X_label = X_cpu.index_select(0, idx)
        # Fixed prototype mode: one centroid per label.
        centroids = F.normalize(X_label.mean(dim=0, keepdim=True), p=2, dim=1)

        target = torch.zeros(L, dtype=torch.float32)
        target[label] = 1.0
        for anc in ancestors.get(label, []):
            target[int(anc)] = 1.0
        X_out.append(centroids)
        Y_out.append(target.unsqueeze(0))

    if not X_out:
        return X_cpu, Y_cpu
    return torch.cat(X_out, dim=0), torch.cat(Y_out, dim=0)

def prepare_memory_inputs(
    X: torch.Tensor,
    Y: torch.Tensor,
    hd
) -> Tuple[torch.Tensor, torch.Tensor]:
    return build_memory_prototypes(X, Y, hd)

def tune_fusion_parameters(
    enc: SharedEncoder,
    clf: DualBranchHierClassifier,
    mem: SemanticMemory,
    engine: InferenceEngine,
    tokens: Dict[str, torch.Tensor],
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    label_tokens: Optional[Dict[str, torch.Tensor]] = None,
    label_levels: Optional[List[int]] = None,
) -> Tuple[float, float, Optional[Dict[int, float]], float, float, float, int]:
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if not fusion_on:
        return cfg.eta, cfg.delta, None, -1.0, -1.0, -1.0, cfg.top_b
    X_va_enc = encode_with_encoder(enc, tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"]
    best_eta = cfg.eta
    best_delta = cfg.delta
    best_delta_levels: Optional[Dict[int, float]] = None
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    best_top_b = cfg.top_b
    top_b_candidates = get_top_b_candidates(cfg)
    eta_candidates = list(getattr(cfg, "eta_candidates", None) or [cfg.eta])
    if cfg.eta not in eta_candidates:
        eta_candidates.append(cfg.eta)
    delta_candidates = list(getattr(cfg, "delta_candidates", None) or [cfg.delta])
    if cfg.delta not in delta_candidates:
        delta_candidates.append(cfg.delta)
    use_level = normalize_delta_mode(cfg) == "level" and label_levels is not None
    if use_level:
        for top_b in top_b_candidates:
            with torch.no_grad():
                s_mem_va = mem.batch_query(X_va_dev, top_b=top_b)
            for eta in eta_candidates:
                s_fused = engine.fuse_scores(s_mem_va, p_cls_va, eta=eta)
                delta_map, score, micro, macro = tune_level_delta_map(
                    s_fused, Y_val, cfg, label_levels, topk=cfg.topk
                )
                if score > best_score:
                    best_score = score
                    best_micro = micro
                    best_macro = macro
                    best_eta = eta
                    best_delta = float(cfg.delta)
                    best_delta_levels = delta_map
                    best_top_b = int(top_b)
    else:
        y_true_va = (Y_val.cpu().numpy() > 0.5).astype(np.int32)
        for top_b in top_b_candidates:
            with torch.no_grad():
                s_mem_va = mem.batch_query(X_va_dev, top_b=top_b)
            for eta in eta_candidates:
                for delta_val in delta_candidates:
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
    return best_eta, best_delta, best_delta_levels, best_score, best_micro, best_macro, best_top_b

def tune_memory_only_parameters(
    enc: SharedEncoder,
    mem: SemanticMemory,
    tokens: Dict[str, torch.Tensor],
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    label_levels: Optional[List[int]] = None,
) -> Tuple[float, Optional[Dict[int, float]], float, float, float, int]:
    X_va_enc = encode_with_encoder(enc, tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)

    best_delta = float(cfg.delta)
    best_delta_levels: Optional[Dict[int, float]] = None
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    best_top_b = int(cfg.top_b)

    top_b_candidates = get_top_b_candidates(cfg)
    with torch.no_grad():
        for top_b in top_b_candidates:
            s_mem_va = mem.batch_query(X_va_dev, top_b=top_b)
            delta_val, delta_levels, score, micro, macro = tune_memory_only_delta(
                s_mem_va,
                Y_val,
                cfg,
                label_levels=label_levels,
                topk=None,
            )
            if score > best_score:
                best_delta = float(delta_val)
                best_delta_levels = delta_levels
                best_score = score
                best_micro = micro
                best_macro = macro
                best_top_b = int(top_b)
    return best_delta, best_delta_levels, best_score, best_micro, best_macro, best_top_b

# -----------------------------
# Main
# -----------------------------
def main(cfg: TrainConfig, summary: Optional[List[Dict[str, float]]] = None, scenario_name: Optional[str] = None) -> Optional[Dict[str, float]]:
    ablation_mode = normalize_ablation_mode(getattr(cfg, "run_ablation", "off"))
    run_ablation = ablation_mode != "off"
    if run_ablation:
        summary = summary if summary is not None else []
        result_path = os.path.join(cfg.workdir, "ablation_result.txt")
        if summary == []:
            init_ablation_result_txt(result_path)
        scenarios: List[Tuple[str, Dict[str, object]]] = []

        if run_ablation:
            scenarios.extend([
                ("global_only", {"use_memory": False, "use_local_branch": False, "use_global_branch": True}),
                ("local_only", {"use_memory": False, "use_local_branch": True, "use_global_branch": False}),
                ("memory_only", {"use_memory": True, "use_local_branch": False, "use_global_branch": False}),
                ("global_local", {"use_memory": False, "use_local_branch": True, "use_global_branch": True}),
                ("all", {"use_memory": True, "use_local_branch": True, "use_global_branch": True}),
            ])

        auto_tune_params = bool(getattr(cfg, "auto_tune_params", True))
        modes = [ablation_mode]
        if "fixed" in modes:
            auto_tune_params = False
        elif "best" in modes:
            auto_tune_params = True

        for name, overrides in scenarios:
            sub_cfg = replace(
                cfg,
                **overrides,
                run_ablation="off",
                auto_tune_params=auto_tune_params,
                workdir=os.path.join(cfg.workdir, name)
            )
            tune_label = "on" if auto_tune_params else "off"
            print(f"\n[Ablation] Running scenario: {name} with {overrides} (tune_params={tune_label})")
            res = main(sub_cfg, summary=summary, scenario_name=name)
            if res is not None:
                summary.append(res)
                append_ablation_result_txt(result_path, res)
        # Print summary table
        if summary:
            print("\n[Ablation Summary]")
            summary_sorted = summary
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
    print(f"Using device: {device}")

    # 1) Load hierarchy and dataset
    root_names = parse_root_label_names(getattr(cfg, "root_label_name", "Root"))
    if bool(getattr(cfg, "exclude_root_label", False)):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as f:
            hjson = json.load(f)
        hjson = promote_named_roots(hjson, root_names)
        hd = parse_label_hierarchy(hjson)
    else:
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
    if bool(getattr(cfg, "exclude_root_label", False)):
        all_label_lists = [strip_root_label(labs, root_names) for labs in all_label_lists]
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
    enc = SharedEncoder(build_encoder_config(cfg, device_str))
    print("[Stage] Tokenizing train/test/label texts...")
    label_descs = build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
    train_tokens = tokenize_texts(enc.tokenizer, tr_texts, cfg.max_len)
    test_tokens = tokenize_texts(enc.tokenizer, te_texts, cfg.max_len)
    label_tokens = tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
    del enc

    # 4) M3: Classifier + losses (M2 memory will be refreshed with current encoder later)
    level_slices = make_level_slices(hd.levels)
    edges_pc = [(int(p), int(c)) for (p, c) in hd.edges_parent_child]
    hierarchy_obj = Hierarchy(num_labels=L, ancestors=hd.ancestors)

    memory_backend = "faiss_ip"
    print(f"[Memory] backend={memory_backend}")
    mem_cfg = MemoryConfig(
        backend=memory_backend,
        top_b=cfg.top_b,
        tau_mem=cfg.tau_mem,
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

    fold_splits = make_k_folds_indices(Y_tr_full.cpu().numpy(), k=4, seed=cfg.seed + 1)

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
    delta_levels_final = best_fold.get("last_tuned_delta_levels", None)
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
        X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr_full, hd)
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
        label_levels=label_levels,
        delta_levels_override=delta_levels_final,
    )

    y_true_te = (Y_te.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_te = pred_te.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_te, y_pred_te)
    macro_all = macro_f1(y_true_te, y_pred_te)
    print("[TEST] Per-label metrics:")
    per_label_report(y_true_te, y_pred_te, hd.id2label)
    print(f"[Final Tuning] Using eta={eta_final:.2f}, delta={delta_final:.2f}, rho={rho_final:.2f}, top_b={top_b_final} derived from best fold. "
          f"(rho=標籤/樣本參數, eta=記憶/分類融合參數, delta=二值化閾值)")
    if normalize_delta_mode(cfg) == "level" and delta_levels_final:
        levels_sorted = sorted(delta_levels_final.items(), key=lambda x: x[0])
        levels_str = ", ".join([f"L{lvl}={val:.2f}" for lvl, val in levels_sorted])
        print(f"[Final Tuning] Per-level delta: {levels_str}")
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")

    export_dir = "outputs"
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "test_result.xlsx")
    rows = []
    id2label = {int(k): v for k, v in hd.id2label.items()}
    texts = df_test[cfg.text_col].astype(str).tolist()
    for i in range(len(texts)):
        true_ids = np.where(y_true_te[i] > 0)[0].tolist()
        pred_ids = np.where(y_pred_te[i] > 0)[0].tolist()
        true_labels = [id2label.get(t, str(t)) for t in true_ids]
        pred_labels = [id2label.get(p, str(p)) for p in pred_ids]
        fp = [id2label.get(p, str(p)) for p in pred_ids if p not in set(true_ids)]
        fn = [id2label.get(t, str(t)) for t in true_ids if t not in set(pred_ids)]
        rows.append({
            "text": texts[i],
            "label": "; ".join(true_labels),
            "pred_label": "; ".join(pred_labels),
            "false_positive": "; ".join(fp),
            "false_negative": "; ".join(fn),
        })
    df_out = pd.DataFrame(rows)
    try:
        df_out.to_excel(export_path, index=False)
        print(f"[Save] Test predictions saved to {export_path}")
    except Exception as exc:
        fallback = os.path.splitext(export_path)[0] + ".csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[Warn] Failed to write xlsx: {exc}. Saved CSV instead: {fallback}")

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
    enc = SharedEncoder(build_encoder_config(cfg, device_str))
    clf_cfg = ClassifierConfig(
        hidden_size=enc.hidden_size,
        level_sizes=hd.level_sizes,
        level_slices=level_slices,
        dropout=cfg.dropout,
        global_hidden_ratio=cfg.global_hidden_ratio if cfg.global_hidden_ratio is not None else cfg.hidden_ratio,
        local_head_hidden_ratio=cfg.local_head_hidden_ratio if cfg.local_head_hidden_ratio is not None else cfg.hidden_ratio,
        local_num_heads=cfg.local_num_heads,
        fusion_hidden_ratio=cfg.fusion_hidden_ratio if cfg.fusion_hidden_ratio is not None else cfg.hidden_ratio,
        global_dropout=cfg.global_dropout,
        local_dropout=cfg.local_dropout,
        use_global_branch=cfg.use_global_branch,
        use_local_branch=cfg.use_local_branch,
        device=device_str,
        loss=LossConfig(
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            use_bce_loss=True,
            path_on_local=False,
            inbatch_tau=cfg.cl_tau,
            weight_cl=1.0,
            use_cl_loss=cfg.use_cl_loss,
            use_path_loss=cfg.use_path_loss,
            weight_path=cfg.weight_path,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    comb = JointLossCombiner(clf_cfg).to(device)
    engine = InferenceEngine(
        InferenceConfig(eta=cfg.eta, delta=cfg.delta, topk=cfg.topk, device=device_str),
        hierarchy_obj
    )

    train_tokens = subset_tokens(train_tokens_full, train_indices)
    val_tokens = subset_tokens(train_tokens_full, val_indices)
    if bool(getattr(cfg, "cache_tokens_on_gpu", False)) and device.type == "cuda":
        train_tokens = move_tokens_to_device(train_tokens, device)
        val_tokens = move_tokens_to_device(val_tokens, device)
    Y_tr = Y_train_full.index_select(0, torch.tensor(train_indices, dtype=torch.long))
    Y_va = Y_train_full.index_select(0, torch.tensor(val_indices, dtype=torch.long))
    label_state = init_label_loss_state(cfg, hd, Y_tr, label_tokens)
    Y_tr_dev = Y_tr.to(device)

    tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100 = build_tail_level_masks(Y_tr)
    sample_weights = compute_sample_weights(
        Y_tr,
        tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100,
        cfg.tail_weight_q0_25, cfg.tail_weight_q25_50, cfg.tail_weight_q50_75, cfg.tail_weight_q75_100,
        label_levels, cfg.level_weight_scale
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count_stage2 = max(0, int(cfg.weighted_extra_ratio * Ntr))
    extra_steps_stage2 = math.ceil(extra_sample_count_stage2 / B) if extra_sample_count_stage2 > 0 else 0
    steps_per_epoch_stage2 = base_steps + extra_steps_stage2

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
    last_tuned_delta_levels: Optional[Dict[int, float]] = None
    last_tuned_rho = cfg.rho
    last_tuned_top_b = cfg.top_b
    auto_tune_params = bool(getattr(cfg, "auto_tune_params", True))

    # ---------------- Stage 2: Classifier / CL-only training ----------------
    configure_cl_for_stage2(comb, cfg, classifier_on=cls_on)

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

    if cls_on:
        assert clf is not None
        param_groups = [
            {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr
            *build_classifier_param_groups(clf, cfg),
        ]
        trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
        train_log_tag = "Cls"
    else:
        param_groups = [
            {"params": enc.parameters(), "lr": cfg.contrast_lr},
        ]
        trainable_params_cls = list(enc.parameters())
        train_log_tag = "MemOnly-CL"
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
        maybe_refresh_label_cache(label_state, enc, cfg, device, ep)
        enc.train()
        if clf is not None:
            clf.train()
        running = {"bce": 0.0, "sample_cl_loss": 0.0, "hnm_cl_loss": 0.0, "path": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_training_indices_common(
            cfg=cfg,
            enc=enc,
            clf=clf,
            comb=comb,
            train_tokens=train_tokens,
            Y_tr=Y_tr_dev,
            batch_indices=base_indices,
            B=B,
            device=device,
            num_labels=num_labels,
            edges_pc=edges_pc,
            trainable_params=trainable_params_cls,
            opt=opt,
            scheduler=scheduler,
            running=running,
            label_state=label_state,
        )

        if extra_sample_count_stage2 > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage2, replacement=True).tolist()
            run_training_indices_common(
                cfg=cfg,
                enc=enc,
                clf=clf,
                comb=comb,
                train_tokens=train_tokens,
                Y_tr=Y_tr_dev,
                batch_indices=extra_indices,
                B=B,
                device=device,
                num_labels=num_labels,
                edges_pc=edges_pc,
                trainable_params=trainable_params_cls,
                opt=opt,
                scheduler=scheduler,
                running=running,
                label_state=label_state,
            )

        print(f"[{fold_name} | {train_log_tag} Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} sample_cl_loss={running['sample_cl_loss']/Ntr:.4f} "
              f"hnm_cl_loss={running['hnm_cl_loss']/Ntr:.4f} path={running['path']/Ntr:.4f}")

        if cls_on and ep % 4 == 0:
            eval_classifier_only(str(ep))
            last_cls_eval_ep = ep
        final_epoch = ep

        avg_loss = running["total"] / Ntr
        stale = 0 if avg_loss < best_loss - 1e-4 else stale + 1
        best_loss = min(best_loss, avg_loss)
        if stale >= patience:
            print(f"[{fold_name}] Early stopping triggered after {patience} stale epochs.")
            break

    if cls_on and final_epoch > 0 and last_cls_eval_ep != final_epoch:
        eval_classifier_only(str(final_epoch))

    # After classifier training, tune memory fusion once on val set
    enc.eval()
    if clf is not None:
        clf.eval()

    fusion_on = cfg.use_memory and cls_on
    mem = None
    tuned_eta = cfg.eta
    tuned_delta = cfg.delta
    tuned_delta_levels: Optional[Dict[int, float]] = None
    tuned_rho = cfg.rho
    tuned_top_b = cfg.top_b
    tuned_score = -1.0
    if cfg.use_memory:
        Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        X_mem_base, Y_mem_base = prepare_memory_inputs(X_tr_mem, Y_tr, hd)
        if auto_tune_params:
            rho_candidates = getattr(cfg, "rho_candidates", None) or [cfg.rho]
            rho_candidates = list(rho_candidates)
            if cfg.rho not in rho_candidates:
                rho_candidates.append(cfg.rho)

            best_mem = None
            best_rho = None
            best_eta = tuned_eta
            best_delta = tuned_delta
            best_delta_levels: Optional[Dict[int, float]] = None
            best_top_b = tuned_top_b
            best_score = -1.0

            for rho_val in rho_candidates:
                mem_tmp = SemanticMemory(mem_cfg)
                mem_tmp.build(X_mem_base, Z_eval, Y_mem_base, rho=rho_val)

                if fusion_on:
                    eta_val, delta_val, delta_levels, score, micro, macro, top_b_val = tune_fusion_parameters(
                        enc, clf, mem_tmp, engine, val_tokens, Y_va, cfg, device,
                        label_tokens=label_tokens, label_levels=label_levels
                    )
                else:
                    eta_val = float(cfg.eta)
                    delta_val, delta_levels, score, micro, macro, top_b_val = tune_memory_only_parameters(
                        enc, mem_tmp, val_tokens, Y_va, cfg, device, label_levels=label_levels
                    )

                if best_mem is None or score > best_score:
                    best_score = score
                    best_mem = mem_tmp
                    best_rho = rho_val
                    best_eta = eta_val
                    best_delta = delta_val
                    best_delta_levels = delta_levels
                    best_top_b = top_b_val

            mem = best_mem
            tuned_rho = best_rho if best_rho is not None else cfg.rho
            tuned_eta = best_eta
            tuned_delta = best_delta
            tuned_delta_levels = best_delta_levels
            tuned_top_b = best_top_b
            tuned_score = best_score
        else:
            mem = SemanticMemory(mem_cfg)
            mem.build(X_mem_base, Z_eval, Y_mem_base, rho=cfg.rho)
            tuned_rho = float(cfg.rho)
            tuned_eta = float(cfg.eta)
            tuned_delta = float(cfg.delta)
            tuned_top_b = int(cfg.top_b)
            tuned_delta_levels = None

    if cfg.use_memory and mem is not None:
        last_tuned_rho = tuned_rho
        last_tuned_top_b = tuned_top_b
        last_tuned_delta = tuned_delta
        last_tuned_delta_levels = tuned_delta_levels

    if fusion_on and mem is not None:
        last_tuned_eta = tuned_eta
        engine.cfg.eta = tuned_eta
        engine.cfg.delta = tuned_delta

    X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"] if clf is not None else torch.zeros(X_va_dev.size(0), int(sum(hd.level_sizes)), device=device)
        s_mem_va = mem.batch_query(X_va_dev, top_b=tuned_top_b) if mem is not None else torch.zeros_like(p_cls_va)
    if not cfg.use_memory:
        if auto_tune_params:
            tuned_delta, tuned_delta_levels, tuned_score, _, _ = tune_classifier_only_delta(
                p_cls_va, Y_va, cfg, label_levels=label_levels, topk=None
            )
        else:
            tuned_delta = float(cfg.delta)
            tuned_score = -1.0
            tuned_delta_levels = None
        last_tuned_delta = tuned_delta
        last_tuned_delta_levels = tuned_delta_levels
    pred_va = predict_with_strategy(
        s_mem=s_mem_va,
        p_cls=p_cls_va,
        engine=engine,
        cfg=cfg,
        eta_override=tuned_eta,
        delta_override=tuned_delta,
        label_levels=label_levels,
        delta_levels_override=tuned_delta_levels,
    )
    y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_va = pred_va.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_va, y_pred_va)
    macro_all = macro_f1(y_true_va, y_pred_va)
    best_score = macro_all if val_metric == "macro" else micro
    best_val_micro = micro
    best_val_macro = macro_all
    if fusion_on and mem is not None:
        fusion_info = f"(rho={tuned_rho:.2f}, eta={tuned_eta:.2f}, delta={tuned_delta:.2f}, top_b={tuned_top_b})"
    elif cfg.use_memory and mem is not None:
        fusion_info = f"(memory_only rho={tuned_rho:.2f}, delta={tuned_delta:.2f}, top_b={tuned_top_b})"
    else:
        fusion_info = "(memory off)"
    print(f"[{fold_name}] VAL (post-train tuning) micro-F1={micro:.4f}  macro-F1={macro_all:.4f}  {fusion_info}")

    torch.save({
        "encoder_state": enc.state_dict(),
        "classifier_state": (clf.state_dict() if clf is not None else None),
        "clf_cfg": clf_cfg.__dict__,
        "val_micro_f1": micro,
        "epoch": cfg.classifier_epochs,
        "rho": tuned_rho,
        "eta": tuned_eta,
        "delta": tuned_delta,
        "delta_levels": tuned_delta_levels,
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
        "last_tuned_delta_levels": last_tuned_delta_levels,
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
    enc = SharedEncoder(build_encoder_config(cfg, device_str))
    clf_cfg = ClassifierConfig(
        hidden_size=enc.hidden_size,
        level_sizes=hd.level_sizes,
        level_slices=level_slices,
        dropout=cfg.dropout,
        global_hidden_ratio=cfg.global_hidden_ratio if cfg.global_hidden_ratio is not None else cfg.hidden_ratio,
        local_head_hidden_ratio=cfg.local_head_hidden_ratio if cfg.local_head_hidden_ratio is not None else cfg.hidden_ratio,
        local_num_heads=cfg.local_num_heads,
        fusion_hidden_ratio=cfg.fusion_hidden_ratio if cfg.fusion_hidden_ratio is not None else cfg.hidden_ratio,
        global_dropout=cfg.global_dropout,
        local_dropout=cfg.local_dropout,
        use_global_branch=cfg.use_global_branch,
        use_local_branch=cfg.use_local_branch,
        device=device_str,
        loss=LossConfig(
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            use_bce_loss=True,
            path_on_local=False,
            inbatch_tau=cfg.cl_tau,
            weight_cl=1.0,
            use_cl_loss=cfg.use_cl_loss,
            use_path_loss=cfg.use_path_loss,
            weight_path=cfg.weight_path,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    comb = JointLossCombiner(clf_cfg).to(device)

    train_tokens = {k: v.to(device) for k, v in train_tokens_full.items()}
    Y_tr = Y_train_full.to(device)
    label_state = init_label_loss_state(cfg, hd, Y_train_full, label_tokens)

    tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100 = build_tail_level_masks(Y_tr)
    sample_weights = compute_sample_weights(
        Y_tr,
        tail_mask_q0_25, tail_mask_q25_50, tail_mask_q50_75, tail_mask_q75_100,
        cfg.tail_weight_q0_25, cfg.tail_weight_q25_50, cfg.tail_weight_q50_75, cfg.tail_weight_q75_100,
        label_levels, cfg.level_weight_scale
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count_stage2 = max(0, int(cfg.weighted_extra_ratio * Ntr))
    extra_steps_stage2 = math.ceil(extra_sample_count_stage2 / B) if extra_sample_count_stage2 > 0 else 0
    steps_per_epoch_stage2 = base_steps + extra_steps_stage2

    num_labels = int(sum(hd.level_sizes))


    # ---------------- Stage 2: Classifier / CL-only training ----------------
    configure_cl_for_stage2(comb, cfg, classifier_on=cls_on)

    if cls_on:
        assert clf is not None
        trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
        param_groups = [
            {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr
            *build_classifier_param_groups(clf, cfg),
        ]
        train_log_tag = "Cls"
    else:
        trainable_params_cls = list(enc.parameters())
        param_groups = [
            {"params": enc.parameters(), "lr": cfg.contrast_lr},
        ]
        train_log_tag = "MemOnly-CL"
    total_steps_cls = max(1, steps_per_epoch_stage2 * cfg.classifier_epochs)
    warmup_steps_cls = int(cfg.warmup_ratio * total_steps_cls)
    opt = torch.optim.AdamW(param_groups, lr=cfg.classifier_lr)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps_cls,
        num_training_steps=total_steps_cls
    )

    for ep in range(1, cfg.classifier_epochs + 1):
        maybe_refresh_label_cache(label_state, enc, cfg, device, ep)
        enc.train()
        if clf is not None:
            clf.train()
        running = {"bce": 0.0, "sample_cl_loss": 0.0, "hnm_cl_loss": 0.0, "path": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_training_indices_common(
            cfg=cfg,
            enc=enc,
            clf=clf,
            comb=comb,
            train_tokens=train_tokens,
            Y_tr=Y_tr,
            batch_indices=base_indices,
            B=B,
            device=device,
            num_labels=num_labels,
            edges_pc=edges_pc,
            trainable_params=trainable_params_cls,
            opt=opt,
            scheduler=scheduler,
            running=running,
            label_state=label_state,
        )

        if extra_sample_count_stage2 > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count_stage2, replacement=True).tolist()
            run_training_indices_common(
                cfg=cfg,
                enc=enc,
                clf=clf,
                comb=comb,
                train_tokens=train_tokens,
                Y_tr=Y_tr,
                batch_indices=extra_indices,
                B=B,
                device=device,
                num_labels=num_labels,
                edges_pc=edges_pc,
                trainable_params=trainable_params_cls,
                opt=opt,
                scheduler=scheduler,
                running=running,
                label_state=label_state,
            )

        print(f"[Full Train | {train_log_tag} Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} sample_cl_loss={running['sample_cl_loss']/Ntr:.4f} "
              f"hnm_cl_loss={running['hnm_cl_loss']/Ntr:.4f} path={running['path']/Ntr:.4f}")

    return enc, clf, clf_cfg

if __name__ == "__main__":
    main(TrainConfig())

