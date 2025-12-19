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
    contrast_epochs: int = 7              # stage 1: contrastive/align pretrain epochs
    classifier_epochs: int = 15           # stage 2: classifier training epochs
    contrast_lr: float = 3e-5             # encoder lr for stage 1 (also used to finetune encoder in stage 2)
    classifier_lr: float = 1e-3           # classifier head lr for stage 2
    seed: int = 42
    warmup_ratio: float = 0.15

    # Memory (M2)
    top_b: int = 200
    temperature: float = 0.04 #論文配置
    lambda_label: float = 0.5
    lambda_candidates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Classifier (M3)
    dropout: float = 0.1
    focal_alpha: float = 0.7
    focal_gamma: float = 0.0

    # path loss
    use_path_loss: bool = True
    weight_path: float = 0.25

    # label loss
    use_label_loss: bool = True
    weight_label: float = 0.05
    num_neg: int = 16  # shared for label/align
    tau_label: float = 0.07

    # align loss
    use_align_loss: bool = True
    weight_align: float = 0.25
    tau_align: float = 0.07

    # Sample-sample contrastive
    use_sample_loss: bool = True
    weight_sample_contrast: float = 0.025
    use_sample_projector: bool = False
    tau_sample_contrast: float = 0.07
    num_neg_sample: int = 4
    sample_repeat: int = 2
    sample_queue_size: int = 32
    exclude_same_level_overlap_neg: bool = False
    average_sample_pos_neg_together: bool = False
    use_inverted_pos_index: bool = True   # enable label->sample inverted index for cross-batch positives
    inverted_pos_per_label: int = 1        # positives per active label when using inverted index

    # Fusion (M4)
    gamma: float = 0.5
    threshold: float = 0.5
    topk: Optional[int] = 15
    gamma_candidates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    threshold_candidates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Module switches / ablations
    use_memory: bool = True
    use_local_branch: bool = True   
    use_global_branch: bool = True
    run_ablation: bool = False     # If True, run predefined ablation scenarios
    run_cl_ablation: bool = False  # If True, run CL-loss ablations (align/label/sample)

    # Sampling (tail-aware / level-aware)
    tail_percentile: float = 50.0
    tail_weight: float = 0.4
    level_threshold: int = 4
    level_weight: float = 0.2
    weighted_extra_ratio: float = 0.3

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

def macro_f1_supported(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-F1 only over labels that appear at least once in y_true.
    """
    L = y_true.shape[1]
    f1s = []
    for j in range(L):
        yt = y_true[:, j].astype(bool)
        if yt.sum() == 0:
            continue
        yp = y_pred[:, j].astype(bool)
        tp = (yt & yp).sum()
        fp = ((~yt) & yp).sum()
        fn = (yt & (~yp)).sum()
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0

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
    gamma_override: Optional[float] = None,
    threshold_override: Optional[float] = None,
) -> torch.Tensor:
    """
    Decide prediction path based on available modules.
    - If memory + fusion (auto when memory on): fuse s_mem and p_cls.
    - If memory only: threshold memory scores.
    - Else: threshold classifier scores (global or global+local depending on clf config).
    """
    gamma = gamma_override if gamma_override is not None else cfg.gamma
    threshold = threshold_override if threshold_override is not None else cfg.threshold
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if fusion_on:
        return engine.predict_batch(
            s_mem, p_cls, gamma=gamma, threshold=threshold
        )["y"]
    if cfg.use_memory:
        return (s_mem >= threshold).to(torch.int64)
    return (p_cls >= threshold).to(torch.int64)


def classifier_enabled(cfg: TrainConfig) -> bool:
    return bool(cfg.use_global_branch or cfg.use_local_branch)


def tune_memory_only_threshold(
    s_mem_val: torch.Tensor,
    Y_val: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[float, float]:
    """
    Tune a single global threshold θ for memory-only predictions:
        y_hat = 1[s_mem >= θ]
    Returns (best_threshold, best_micro_f1).
    """
    y_true = (Y_val.detach().cpu().numpy() > 0.5).astype(np.int32)
    candidates = list(getattr(cfg, "threshold_candidates", None) or [cfg.threshold])
    if cfg.threshold not in candidates:
        candidates.append(cfg.threshold)
    # Guard: keep within [0,1]
    candidates = [float(max(0.0, min(1.0, t))) for t in candidates]

    best_thr = float(cfg.threshold)
    best_micro = -1.0
    s_cpu = s_mem_val.detach().cpu()
    for thr in candidates:
        y_pred = (s_cpu >= thr).numpy().astype(np.int32)
        micro = micro_f1(y_true, y_pred)
        if micro > best_micro:
            best_micro = micro
            best_thr = float(thr)
    return best_thr, best_micro


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


def build_inverted_index(Y: torch.Tensor) -> Dict[int, List[int]]:
    """Build label -> sample indices mapping (CPU) for cross-batch positive sampling."""
    Yn = (Y > 0.5).cpu().numpy()
    L = Yn.shape[1]
    inv: Dict[int, List[int]] = {j: [] for j in range(L)}
    for i, row in enumerate(Yn):
        labels = np.nonzero(row)[0]
        for j in labels:
            inv[j].append(i)
    return inv


def sample_cross_batch_positives(
    batch_indices: List[int],
    Y_cpu: np.ndarray,
    inv_index: Dict[int, List[int]],
    per_label: int = 1,
) -> List[int]:
    """For each anchor in batch and each active label, draw up to per_label other samples as positives."""
    picked: List[int] = []
    for idx in batch_indices:
        labels = np.nonzero(Y_cpu[idx])[0]
        for j in labels:
            cand = inv_index.get(j, [])
            if not cand:
                continue
            # avoid picking the anchor itself if possible
            choices = [c for c in cand if c != idx] if len(cand) > 1 else cand
            if not choices:
                continue
            take = min(per_label, len(choices))
            for pick in np.random.choice(choices, size=take, replace=len(choices) < take):
                picked.append(int(pick))
    # Deduplicate to reduce extra forward calls
    return sorted(set(picked))


def to_pos_idx_list(Y: torch.Tensor, label_levels: List[int], min_level: int = 3) -> List[List[int]]:
    idxs: List[List[int]] = []
    Yn = (Y > 0.5).cpu().numpy()
    for row in Yn:
        all_indices = np.nonzero(row)[0].tolist()
        filtered = [j for j in all_indices if label_levels[j] >= min_level]
        idxs.append(filtered if filtered else all_indices)
    return idxs


def compute_descendants(children_map: Dict[int, List[int]], L: int) -> Dict[int, set]:
    descendants = {i: set() for i in range(L)}
    for node in range(L):
        stack = list(children_map.get(node, []))
        seen = set()
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            descendants[node].add(child)
            stack.extend(children_map.get(child, []))
    return descendants


def build_tail_level_masks(
    Y: torch.Tensor,
    tail_percentile: float,
    level_threshold: int,
    label_levels: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = Y.sum(dim=0).cpu().numpy()
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        tail_thresh = 0.0
    else:
        tail_percentile = min(max(tail_percentile, 0.0), 100.0)
        tail_thresh = float(np.percentile(nonzero, tail_percentile))
    tail_mask = torch.tensor(counts <= tail_thresh, dtype=torch.bool)
    level_mask = torch.tensor([lvl >= level_threshold for lvl in label_levels], dtype=torch.bool)
    return tail_mask, level_mask


def compute_sample_weights(
    Y: torch.Tensor,
    tail_mask: torch.Tensor,
    level_mask: torch.Tensor,
    tail_weight: float,
    level_weight: float
) -> torch.Tensor:
    weights = torch.ones(Y.size(0), device=Y.device)
    if tail_mask.any():
        tail_hits = (Y[:, tail_mask.to(Y.device)] > 0.5).any(dim=1)
        weights += tail_weight * tail_hits.float()
    if level_mask.any():
        level_hits = (Y[:, level_mask.to(Y.device)] > 0.5).any(dim=1)
        weights += level_weight * level_hits.float()
    return weights


def tune_fusion_parameters(
    enc: SharedEncoder,
    clf: DualBranchHierClassifier,
    mem: SemanticMemory,
    engine: InferenceEngine,
    tokens: Dict[str, torch.Tensor],
    Y_val: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device
) -> Tuple[float, float, float]:
    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if not fusion_on:
        return cfg.gamma, cfg.threshold, -1.0
    X_va_enc = encode_with_encoder(enc, tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"]
        s_mem_va = mem.batch_query(X_va_dev)
    y_true_va = (Y_val.cpu().numpy() > 0.5).astype(np.int32)
    best_gamma = cfg.gamma
    best_threshold = cfg.threshold
    best_micro = -1.0
    for gamma in cfg.gamma_candidates:
        for thr in cfg.threshold_candidates:
            pred = engine.predict_batch(
                s_mem_va, p_cls_va, gamma=gamma, threshold=thr
            )
            y_pred = pred["y"].cpu().numpy().astype(np.int32)
            micro = micro_f1(y_true_va, y_pred)
            if micro > best_micro:
                best_micro = micro
                best_gamma = gamma
                best_threshold = thr
    return best_gamma, best_threshold, best_micro



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
                ("global_local", {"use_memory": False, "use_local_branch": True, "use_global_branch": True}),
                ("memory_only", {"use_memory": True, "use_local_branch": False, "use_global_branch": False}),
                ("memory_global", {"use_memory": True, "use_local_branch": False, "use_global_branch": True}),
                ("memory_local", {"use_memory": True, "use_local_branch": True, "use_global_branch": False}),
                ("all", {"use_memory": True, "use_local_branch": True, "use_global_branch": True}),
            ])

        if getattr(cfg, "run_cl_ablation", False):
            align_w = cfg.weight_align
            label_w = cfg.weight_label
            sample_w = cfg.weight_sample_contrast
            scenarios.extend([
                ("cl_all", {}),
                ("cl_no_align", {"use_align_loss": False, "weight_align": 0.0}),
                ("cl_no_label", {"use_label_loss": False, "weight_label": 0.0}),
                ("cl_no_sample", {"use_sample_loss": False, "weight_sample_contrast": 0.0}),
                ("cl_align_only", {
                    "use_align_loss": True, "weight_align": align_w,
                    "use_label_loss": False, "weight_label": 0.0,
                    "use_sample_loss": False, "weight_sample_contrast": 0.0,
                }),
                ("cl_label_only", {
                    "use_align_loss": False, "weight_align": 0.0,
                    "use_label_loss": True, "weight_label": label_w,
                    "use_sample_loss": False, "weight_sample_contrast": 0.0,
                }),
                ("cl_sample_only", {
                    "use_align_loss": False, "weight_align": 0.0,
                    "use_label_loss": False, "weight_label": 0.0,
                    "use_sample_loss": True, "weight_sample_contrast": sample_w,
                }),
                ("cl_none", {
                    "use_align_loss": False, "weight_align": 0.0,
                    "use_label_loss": False, "weight_label": 0.0,
                    "use_sample_loss": False, "weight_sample_contrast": 0.0,
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
            for item in summary:
                use_memory = bool(item.get("use_memory", True))
                gamma_print = 0.0 if not use_memory else float(item.get("gamma", 0.0))
                print(f"  - {item['scenario']}: gamma={gamma_print:.2f}, threshold={item['threshold']:.2f}, "
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
    ancestors_map = {int(k): [int(a) for a in v] for k, v in hd.ancestors.items()}
    children_src = getattr(hd, "children", {})
    children_map = {int(k): [int(c) for c in v] for k, v in children_src.items()}
    for node in range(L):
        children_map.setdefault(node, [])
        ancestors_map.setdefault(node, [])
    descendants_map = compute_descendants(children_map, L)
    forbid_relatives = {i: set(ancestors_map.get(i, [])) | descendants_map.get(i, set()) for i in range(L)}

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
        lambda_label=cfg.lambda_label,
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
            forbid_relatives=forbid_relatives,
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
        if best_fold is None or result["best_val_micro"] > best_fold["best_val_micro"]:
            best_fold = result

    print("\n[CV] Fold summary:")
    for res in fold_results:
        lam_info = res.get("last_tuned_lambda", None)
        if lam_info is not None:
            print(f"  - {res['fold']}: best val micro-F1={res['best_val_micro']:.4f}, "
                  f"lambda={lam_info:.2f}, checkpoint={res['best_path']}")
        else:
            print(f"  - {res['fold']}: best val micro-F1={res['best_val_micro']:.4f}, "
                  f"checkpoint={res['best_path']}")

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

    gamma_final = best_fold["last_tuned_gamma"] if best_fold.get("last_tuned_gamma") is not None else cfg.gamma
    threshold_final = best_fold["last_tuned_threshold"] if best_fold.get("last_tuned_threshold") is not None else cfg.threshold
    lambda_final = best_fold["last_tuned_lambda"] if best_fold.get("last_tuned_lambda") is not None else cfg.lambda_label
    mem_cfg_final = replace(mem_cfg, lambda_label=lambda_final)

    # ----------------- Retrain on full training data -----------------
    enc, clf, clf_cfg_full = train_full_model(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        same_level_map=same_level_map,
        forbid_relatives=forbid_relatives,
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
        InferenceConfig(gamma=gamma_final, threshold=threshold_final, topk=cfg.topk, device=device_str),
        hierarchy_obj
    )

    mem = None
    if cfg.use_memory:
        Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        mem = SemanticMemory(mem_cfg_final)
        mem.build(X_tr_mem, Z_eval, Y_tr_full.detach().cpu())

    print(f"[Final Tuning] Using gamma={gamma_final:.2f}, threshold={threshold_final:.2f}, lambda={lambda_final:.2f} derived from best fold.")

    X_te_enc = encode_with_encoder(enc, test_tokens, cfg.batch_size, device)
    X_te_dev = X_te_enc.to(device)
    with torch.no_grad():
        if clf is not None:
            p_cls_te = clf(X_te_dev)["p_cls"]
        else:
            p_cls_te = torch.zeros(X_te_dev.size(0), L, device=device)
        s_mem_te = mem.batch_query(X_te_dev) if mem is not None else torch.zeros_like(p_cls_te)
    pred_te = predict_with_strategy(
        s_mem=s_mem_te,
        p_cls=p_cls_te,
        engine=engine,
        cfg=cfg,
        gamma_override=gamma_final,
        threshold_override=threshold_final,
    )

    y_true_te = (Y_te.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_te = pred_te.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_te, y_pred_te)
    macro_all = macro_f1(y_true_te, y_pred_te)
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")
    print("[TEST] Per-label metrics:")
    per_label_report(y_true_te, y_pred_te, hd.id2label)

    # Save artifacts for reproducibility
    if mem is not None:
        mem.save(os.path.join(cfg.workdir, "memory_store"))
    # Save full-train checkpoint for downstream use (encoder + classifier)
    full_ckpt_path = os.path.join(cfg.workdir, "best_model_full.pt")
    torch.save({
        "encoder_state": enc.state_dict(),
        "classifier_state": (clf.state_dict() if clf is not None else None),
        "clf_cfg": clf_cfg_full.__dict__,
        "gamma": gamma_final,
        "threshold": threshold_final,
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
        "gamma": gamma_final,
        "threshold": threshold_final,
        "micro": micro,
        "macro_all": macro_all,
        "use_memory": bool(cfg.use_memory),
    }


def train_single_fold(
    fold_name: str,
    cfg: TrainConfig,
    hd,
    hierarchy_obj: Hierarchy,
    level_slices,
    label_levels,
    same_level_map,
    forbid_relatives,
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
            num_neg=cfg.num_neg,
            use_align_loss=cfg.use_align_loss,
            use_label_loss=cfg.use_label_loss,
            use_path_loss=cfg.use_path_loss,
            weight_label=cfg.weight_label,
            weight_align=cfg.weight_align,
            weight_path=cfg.weight_path,
            weight_sample_contrast=cfg.weight_sample_contrast,
            tau_sample_contrast=cfg.tau_sample_contrast,
            use_sample_projector=cfg.use_sample_projector,
            use_sample_loss=cfg.use_sample_loss,
            num_neg_sample=cfg.num_neg_sample,
            sample_repeat=cfg.sample_repeat,
            sample_queue_size=cfg.sample_queue_size,
            exclude_same_level_overlap_neg=cfg.exclude_same_level_overlap_neg,
            average_sample_pos_neg_together=cfg.average_sample_pos_neg_together,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    label_tokens_device = move_tokens_to_device(label_tokens, device)
    comb = JointLossCombiner(clf_cfg).to(device)
    engine = InferenceEngine(
        InferenceConfig(gamma=cfg.gamma, threshold=cfg.threshold, topk=cfg.topk, device=device_str),
        hierarchy_obj
    )

    train_tokens = subset_tokens(train_tokens_full, train_indices)
    val_tokens = subset_tokens(train_tokens_full, val_indices)
    Y_tr = Y_train_full.index_select(0, torch.tensor(train_indices, dtype=torch.long))
    Y_va = Y_train_full.index_select(0, torch.tensor(val_indices, dtype=torch.long))
    mask_tr = torch.ones_like(Y_tr)

    inv_index = None
    Y_tr_cpu = None
    if cfg.use_inverted_pos_index:
        inv_index = build_inverted_index(Y_tr)
        Y_tr_cpu = (Y_tr > 0.5).cpu().numpy()

    tail_mask, level_mask = build_tail_level_masks(
        Y_tr, cfg.tail_percentile, cfg.level_threshold, label_levels
    )
    sample_weights = compute_sample_weights(
        Y_tr, tail_mask, level_mask, cfg.tail_weight, cfg.level_weight
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count = max(0, int(cfg.weighted_extra_ratio * Ntr))
    extra_steps = math.ceil(extra_sample_count / B) if extra_sample_count > 0 else 0
    steps_per_epoch = base_steps + extra_steps

    pos_tr = to_pos_idx_list(Y_tr, label_levels)
    best_f1 = -1.0
    best_loss = float("inf")
    best_path = os.path.join(cfg.workdir, f"best_model_{fold_name}.pt")
    patience = 3
    stale = 0
    last_tuned_gamma = cfg.gamma
    last_tuned_threshold = cfg.threshold
    last_tuned_lambda = cfg.lambda_label

    def run_indices(batch_indices: List[int], opt, scheduler, running: Dict[str, float], trainable_params: List[torch.nn.Parameter]):
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

            label_kwargs = {
                "input_ids": label_tokens_device["input_ids"],
                "attention_mask": label_tokens_device["attention_mask"],
            }
            if "token_type_ids" in label_tokens_device:
                label_kwargs["token_type_ids"] = label_tokens_device["token_type_ids"]
            Z = enc.forward(**label_kwargs)

            batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long)
            Yb = Y_tr.index_select(0, batch_idx_tensor).to(device)
            mask_b = mask_tr.index_select(0, batch_idx_tensor).to(device)
            pos_b = [pos_tr[i] for i in batch_idx]

            extra_feats = None
            extra_labels = None
            if inv_index is not None and Y_tr_cpu is not None:
                extra_indices = sample_cross_batch_positives(
                    batch_idx, Y_tr_cpu, inv_index, per_label=cfg.inverted_pos_per_label
                )
                if extra_indices:
                    extra_tokens = select_tokens_by_index(train_tokens, extra_indices, device)
                    extra_feats = enc.forward(**extra_tokens)
                    # Y_tr is on CPU in fold training; keep indices on CPU then move labels to device
                    extra_idx_tensor = torch.tensor(extra_indices, dtype=torch.long)
                    extra_labels = Y_tr.index_select(0, extra_idx_tensor).to(device)

            if clf is not None:
                out = clf(h_x)
                p_cls = out["p_cls"]
                p_local = out.get("p_local")
            else:
                L = int(sum(hd.level_sizes))
                p_cls = torch.zeros(h_x.size(0), L, device=device)
                p_local = None

            losses = comb(
                p_cls=p_cls,
                p_local=p_local,
                Y=Yb,
                mask=mask_b,
                h_x=h_x,
                Z=Z,
                pos_indices_per_sample=pos_b,
                edges_parent_child=edges_pc,
                Z_for_label_loss=Z,
                same_level_map=same_level_map,
                label_levels=label_levels,
                forbid_relatives=forbid_relatives,
                extra_pos_feats=extra_feats,
                extra_pos_labels=extra_labels,
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
            running["sample"] += float(losses.get("loss_sample", 0.0)) * bs
            running["total"] += float(loss) * bs

    # ---------------- Stage 1: Contrastive/align pretrain ----------------
    if cfg.contrast_epochs > 0:
        comb.loss.use_bce_loss = False
        comb.loss.use_align_loss = bool(cfg.use_align_loss)
        comb.loss.use_label_loss = bool(cfg.use_label_loss)
        comb.loss.use_sample_loss = bool(cfg.use_sample_loss)
        comb.loss.use_path_loss = False
        comb.loss.weight_align = cfg.weight_align if cfg.use_align_loss else 0.0
        comb.loss.weight_label = cfg.weight_label if cfg.use_label_loss else 0.0
        comb.loss.weight_sample_contrast = cfg.weight_sample_contrast if cfg.use_sample_loss else 0.0
        comb.loss.weight_path = 0.0

        total_steps_contrast = max(1, steps_per_epoch * cfg.contrast_epochs)
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
            running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "sample": 0.0, "total": 0.0}

            base_indices = torch.randperm(Ntr).tolist()
            run_indices(base_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            if extra_sample_count > 0:
                extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count, replacement=True).tolist()
                run_indices(extra_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            print(f"[{fold_name} | Contrast Ep {ep}] train total={running['total']/Ntr:.4f} | "
                  f"bce={running['bce']/Ntr:.4f} align={running['align']/Ntr:.4f} "
                  f"path={running['path']/Ntr:.4f} label={running['label']/Ntr:.4f} "
                  f"sample={running['sample']/Ntr:.4f}")

            # For memory-only (no M3), validate after each contrast epoch and save best encoder.
            if (not cls_on) and cfg.use_memory:
                enc.eval()
                tuned_lambda = cfg.lambda_label
                tuned_threshold = cfg.threshold
                best_micro = -1.0
                best_s_mem_va: Optional[torch.Tensor] = None

                Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
                X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
                X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
                X_va_dev = X_va_enc.to(device)

                lambda_candidates = list(getattr(cfg, "lambda_candidates", None) or [cfg.lambda_label])
                if cfg.lambda_label not in lambda_candidates:
                    lambda_candidates.append(cfg.lambda_label)

                for lam in lambda_candidates:
                    mem_tmp = SemanticMemory(mem_cfg)
                    mem_tmp.build(X_tr_mem, Z_eval, Y_tr.detach().cpu(), lambda_label=lam)
                    with torch.no_grad():
                        s_mem_va = mem_tmp.batch_query(X_va_dev)
                    thr, micro = tune_memory_only_threshold(s_mem_va, Y_va, cfg)
                    if micro > best_micro:
                        best_micro = micro
                        tuned_lambda = float(lam)
                        tuned_threshold = float(thr)
                        best_s_mem_va = s_mem_va.detach()

                y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
                L = int(sum(hd.level_sizes))
                s_for_eval = best_s_mem_va if best_s_mem_va is not None else torch.zeros(X_va_dev.size(0), L, device=X_va_dev.device)
                y_pred_va = (s_for_eval.detach().cpu().numpy() >= tuned_threshold).astype(np.int32)
                micro = micro_f1(y_true_va, y_pred_va)
                macro_all = macro_f1(y_true_va, y_pred_va)
                macro_supported = macro_f1_supported(y_true_va, y_pred_va)
                print(f"[{fold_name} | Contrast Ep {ep}] VAL  micro-F1={micro:.4f}  macro-F1(all)={macro_all:.4f}  "
                      f"macro-F1(supported)={macro_supported:.4f}  (memory_only: lambda={tuned_lambda:.2f}, thr={tuned_threshold:.2f})")

                if micro > best_f1:
                    best_f1 = micro
                    last_tuned_gamma = 1.0
                    last_tuned_threshold = tuned_threshold
                    last_tuned_lambda = tuned_lambda
                    torch.save({
                        "encoder_state": enc.state_dict(),
                        "classifier_state": None,
                        "clf_cfg": clf_cfg.__dict__,
                        "val_micro_f1": best_f1,
                        "epoch": ep,
                        "memory_only": True,
                        "lambda_label": tuned_lambda,
                        "gamma": 1.0,
                        "threshold": tuned_threshold,
                    }, best_path)
                    print(f"  -> saved best checkpoint to {best_path}")
                    stale = 0
                else:
                    stale += 1

                if stale >= patience:
                    print(f"[{fold_name}] Early stopping triggered after {patience} stale epochs.")
                    break

    # Reset sample queue before classifier stage
    comb.sample_queue_feats = None
    comb.sample_queue_labels = None

    # ---------------- Stage 2: Classifier training ----------------
    if not cls_on:
        # If no contrast epochs (or never improved), do a single memory-only validation pass.
        if cfg.use_memory and (best_f1 < 0.0) and (not os.path.exists(best_path)):
            enc.eval()
            Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
            X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
            X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
            X_va_dev = X_va_enc.to(device)

            tuned_lambda = cfg.lambda_label
            tuned_threshold = cfg.threshold
            best_micro = -1.0
            best_s_mem_va: Optional[torch.Tensor] = None

            lambda_candidates = list(getattr(cfg, "lambda_candidates", None) or [cfg.lambda_label])
            if cfg.lambda_label not in lambda_candidates:
                lambda_candidates.append(cfg.lambda_label)

            for lam in lambda_candidates:
                mem_tmp = SemanticMemory(mem_cfg)
                mem_tmp.build(X_tr_mem, Z_eval, Y_tr.detach().cpu(), lambda_label=lam)
                with torch.no_grad():
                    s_mem_va = mem_tmp.batch_query(X_va_dev)
                thr, micro = tune_memory_only_threshold(s_mem_va, Y_va, cfg)
                if micro > best_micro:
                    best_micro = micro
                    tuned_lambda = float(lam)
                    tuned_threshold = float(thr)
                    best_s_mem_va = s_mem_va.detach()

            last_tuned_gamma = 1.0
            last_tuned_threshold = tuned_threshold
            last_tuned_lambda = tuned_lambda

            y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
            L = int(sum(hd.level_sizes))
            s_for_eval = best_s_mem_va if best_s_mem_va is not None else torch.zeros(X_va_dev.size(0), L, device=X_va_dev.device)
            y_pred_va = (s_for_eval.detach().cpu().numpy() >= tuned_threshold).astype(np.int32)
            best_f1 = micro_f1(y_true_va, y_pred_va)
            torch.save({
                "encoder_state": enc.state_dict(),
                "classifier_state": None,
                "clf_cfg": clf_cfg.__dict__,
                "val_micro_f1": best_f1,
                "epoch": 0,
                "memory_only": True,
                "lambda_label": tuned_lambda,
                "gamma": 1.0,
                "threshold": tuned_threshold,
            }, best_path)
            print(f"[{fold_name}] Saved memory-only checkpoint to {best_path} (micro-F1={best_f1:.4f}, lambda={tuned_lambda:.2f}, thr={tuned_threshold:.2f})")

        return {
            "fold": fold_name,
            "best_val_micro": best_f1,
            "best_path": best_path if os.path.exists(best_path) else None,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "last_tuned_gamma": last_tuned_gamma,
            "last_tuned_threshold": last_tuned_threshold,
            "last_tuned_lambda": last_tuned_lambda,
        }

    comb.loss.use_bce_loss = True  # BCE/Focal 強制開
    comb.loss.use_align_loss = False
    comb.loss.use_label_loss = False
    comb.loss.use_sample_loss = False
    comb.loss.use_path_loss = cfg.use_path_loss
    comb.loss.weight_align = 0.0
    comb.loss.weight_label = 0.0
    comb.loss.weight_sample_contrast = 0.0
    comb.loss.weight_path = cfg.weight_path if cfg.use_path_loss else 0.0

    param_groups = [
        {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr matches stage 1
        {"params": clf.parameters(), "lr": cfg.classifier_lr},
    ]
    trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
    total_steps_cls = max(1, steps_per_epoch * cfg.classifier_epochs)
    warmup_steps_cls = int(cfg.warmup_ratio * total_steps_cls)
    opt = torch.optim.AdamW(param_groups, lr=cfg.classifier_lr)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps_cls,
        num_training_steps=total_steps_cls
    )

    for ep in range(1, cfg.classifier_epochs + 1):
        enc.train()
        if clf is not None:
            clf.train()
        running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "sample": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_indices(base_indices, opt, scheduler, running, trainable_params_cls)

        if extra_sample_count > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count, replacement=True).tolist()
            run_indices(extra_indices, opt, scheduler, running, trainable_params_cls)

        print(f"[{fold_name} | Cls Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} align={running['align']/Ntr:.4f} "
              f"path={running['path']/Ntr:.4f} label={running['label']/Ntr:.4f} "
              f"sample={running['sample']/Ntr:.4f}")
        avg_loss = running["total"] / Ntr
        stale = 0 if avg_loss < best_loss - 1e-4 else stale + 1
        best_loss = min(best_loss, avg_loss)
        if stale >= patience:
            print(f"[{fold_name}] Early stopping triggered after {patience} stale epochs.")
            break

    # After classifier training, tune memory fusion once on val set
    enc.eval()
    if clf is not None:
        clf.eval()

    mem = None
    tuned_gamma = cfg.gamma
    tuned_threshold = cfg.threshold
    tuned_lambda = cfg.lambda_label
    tuned_micro = -1.0
    if cfg.use_memory:
        Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        lambda_candidates = getattr(cfg, "lambda_candidates", None) or [cfg.lambda_label]
        lambda_candidates = list(lambda_candidates)
        if cfg.lambda_label not in lambda_candidates:
            lambda_candidates.append(cfg.lambda_label)
        fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)

        best_mem = None
        best_lam = None
        best_gamma = tuned_gamma
        best_threshold = tuned_threshold
        best_micro = -1.0

        for lam in lambda_candidates:
            mem_tmp = SemanticMemory(mem_cfg)
            mem_tmp.build(X_tr_mem, Z_eval, Y_tr.detach().cpu(), lambda_label=lam)

            if fusion_on:
                g, thr, m = tune_fusion_parameters(
                    enc, clf, mem_tmp, engine, val_tokens, Y_va, cfg, device
                )
            else:
                g, thr, m = tuned_gamma, tuned_threshold, -1.0

            if best_mem is None or m > best_micro:
                best_micro = m
                best_mem = mem_tmp
                best_lam = lam
                best_gamma = g
                best_threshold = thr

        mem = best_mem
        tuned_lambda = best_lam if best_lam is not None else cfg.lambda_label
        tuned_gamma = best_gamma
        tuned_threshold = best_threshold
        tuned_micro = best_micro

    fusion_on = cfg.use_memory and (cfg.use_global_branch or cfg.use_local_branch)
    if fusion_on and mem is not None:
        last_tuned_gamma = tuned_gamma
        last_tuned_threshold = tuned_threshold
        last_tuned_lambda = tuned_lambda
        engine.cfg.gamma = tuned_gamma
        engine.cfg.threshold = tuned_threshold

    X_va_enc = encode_with_encoder(enc, val_tokens, cfg.batch_size, device)
    X_va_dev = X_va_enc.to(device)
    with torch.no_grad():
        p_cls_va = clf(X_va_dev)["p_cls"] if clf is not None else torch.zeros(X_va_dev.size(0), int(sum(hd.level_sizes)), device=device)
        s_mem_va = mem.batch_query(X_va_dev) if mem is not None else torch.zeros_like(p_cls_va)
    pred_va = predict_with_strategy(
        s_mem=s_mem_va,
        p_cls=p_cls_va,
        engine=engine,
        cfg=cfg,
        gamma_override=tuned_gamma,
        threshold_override=tuned_threshold,
    )
    y_true_va = (Y_va.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_va = pred_va.cpu().numpy().astype(np.int32)

    micro = micro_f1(y_true_va, y_pred_va)
    macro_all = macro_f1(y_true_va, y_pred_va)
    macro_supported = macro_f1_supported(y_true_va, y_pred_va)
    fusion_info = (f"(lambda={tuned_lambda:.2f}, gamma={tuned_gamma:.2f}, thr={tuned_threshold:.2f}, tuned_micro={tuned_micro:.4f})"
                   if (fusion_on and mem is not None) else "(fusion off)")
    print(f"[{fold_name}] VAL (post-train tuning) micro-F1={micro:.4f}  macro-F1(all)={macro_all:.4f}  macro-F1(supported)={macro_supported:.4f}  {fusion_info}")

    best_f1 = micro
    torch.save({
        "encoder_state": enc.state_dict(),
        "classifier_state": clf.state_dict(),
        "clf_cfg": clf_cfg.__dict__,
        "val_micro_f1": best_f1,
        "epoch": cfg.classifier_epochs,
        "lambda_label": tuned_lambda,
        "gamma": tuned_gamma,
        "threshold": tuned_threshold,
    }, best_path)
    print(f"  -> saved checkpoint to {best_path}")

    return {
        "fold": fold_name,
        "best_val_micro": best_f1,
        "best_path": best_path if os.path.exists(best_path) else None,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "last_tuned_gamma": last_tuned_gamma,
        "last_tuned_threshold": last_tuned_threshold,
        "last_tuned_lambda": last_tuned_lambda,
    }


def train_full_model(
    cfg: TrainConfig,
    hd,
    hierarchy_obj: Hierarchy,
    level_slices,
    label_levels,
    same_level_map,
    forbid_relatives,
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
            num_neg=cfg.num_neg,
            use_align_loss=cfg.use_align_loss,
            use_label_loss=cfg.use_label_loss,
            use_path_loss=cfg.use_path_loss,
            weight_label=cfg.weight_label,
            weight_align=cfg.weight_align,
            weight_path=cfg.weight_path,
            weight_sample_contrast=cfg.weight_sample_contrast,
            tau_sample_contrast=cfg.tau_sample_contrast,
            use_sample_projector=cfg.use_sample_projector,
            use_sample_loss=cfg.use_sample_loss,
            num_neg_sample=cfg.num_neg_sample,
            sample_repeat=cfg.sample_repeat,
            sample_queue_size=cfg.sample_queue_size,
            exclude_same_level_overlap_neg=cfg.exclude_same_level_overlap_neg,
            average_sample_pos_neg_together=cfg.average_sample_pos_neg_together,
        )
    )
    clf = DualBranchHierClassifier(clf_cfg).to(device) if cls_on else None
    comb = JointLossCombiner(clf_cfg).to(device)
    label_tokens_device = move_tokens_to_device(label_tokens, device)

    train_tokens = {k: v.to(device) for k, v in train_tokens_full.items()}
    Y_tr = Y_train_full.to(device)
    mask_tr = torch.ones_like(Y_tr)

    inv_index = None
    Y_tr_cpu = None
    if cfg.use_inverted_pos_index:
        inv_index = build_inverted_index(Y_train_full)
        Y_tr_cpu = (Y_train_full > 0.5).cpu().numpy()

    tail_mask, level_mask = build_tail_level_masks(
        Y_tr, cfg.tail_percentile, cfg.level_threshold, label_levels
    )
    sample_weights = compute_sample_weights(
        Y_tr, tail_mask, level_mask, cfg.tail_weight, cfg.level_weight
    ).clamp_min(1e-6)
    weight_tensor = (sample_weights / sample_weights.sum()).to(torch.float32)

    Ntr = train_tokens["input_ids"].size(0)
    B = min(256, cfg.batch_size)
    base_steps = math.ceil(Ntr / B)
    extra_sample_count = max(0, int(cfg.weighted_extra_ratio * Ntr))
    extra_steps = math.ceil(extra_sample_count / B) if extra_sample_count > 0 else 0
    steps_per_epoch = base_steps + extra_steps

    pos_tr = to_pos_idx_list(Y_tr, label_levels)

    def run_indices(batch_indices: List[int], opt, scheduler, running: Dict[str, float], trainable_params: List[torch.nn.Parameter]):
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

            label_kwargs = {
                "input_ids": label_tokens_device["input_ids"],
                "attention_mask": label_tokens_device["attention_mask"],
            }
            if "token_type_ids" in label_tokens_device:
                label_kwargs["token_type_ids"] = label_tokens_device["token_type_ids"]
            Z = enc.forward(**label_kwargs)

            Yb = Y_tr.index_select(0, idx_tensor)
            mask_b = mask_tr.index_select(0, idx_tensor)
            pos_b = [pos_tr[i] for i in batch_idx]

            extra_feats = None
            extra_labels = None
            if inv_index is not None and Y_tr_cpu is not None:
                extra_indices = sample_cross_batch_positives(
                    batch_idx, Y_tr_cpu, inv_index, per_label=cfg.inverted_pos_per_label
                )
                if extra_indices:
                    extra_tokens = select_tokens_by_index(train_tokens, extra_indices, device)
                    extra_feats = enc.forward(**extra_tokens)
                    extra_idx_tensor = torch.tensor(extra_indices, dtype=torch.long, device=device)
                    extra_labels = Y_tr.index_select(0, extra_idx_tensor)

            if clf is not None:
                out = clf(h_x)
                p_cls = out["p_cls"]
                p_local = out.get("p_local")
            else:
                L = int(sum(hd.level_sizes))
                p_cls = torch.zeros(h_x.size(0), L, device=device)
                p_local = None

            losses = comb(
                p_cls=p_cls,
                p_local=p_local,
                Y=Yb,
                mask=mask_b,
                h_x=h_x,
                Z=Z,
                pos_indices_per_sample=pos_b,
                edges_parent_child=edges_pc,
                Z_for_label_loss=Z,
                same_level_map=same_level_map,
                label_levels=label_levels,
                forbid_relatives=forbid_relatives,
                extra_pos_feats=extra_feats,
                extra_pos_labels=extra_labels,
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
            running["sample"] += float(losses.get("loss_sample", 0.0)) * bs

    # ---------------- Stage 1: Contrastive/align pretrain ----------------
    if cfg.contrast_epochs > 0:
        comb.loss.use_bce_loss = False
        comb.loss.use_align_loss = bool(cfg.use_align_loss)
        comb.loss.use_label_loss = bool(cfg.use_label_loss)
        comb.loss.use_sample_loss = bool(cfg.use_sample_loss)
        comb.loss.use_path_loss = False
        comb.loss.weight_align = cfg.weight_align if cfg.use_align_loss else 0.0
        comb.loss.weight_label = cfg.weight_label if cfg.use_label_loss else 0.0
        comb.loss.weight_sample_contrast = cfg.weight_sample_contrast if cfg.use_sample_loss else 0.0
        comb.loss.weight_path = 0.0

        total_steps_contrast = max(1, steps_per_epoch * cfg.contrast_epochs)
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
            running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "sample": 0.0, "total": 0.0}

            base_indices = torch.randperm(Ntr).tolist()
            run_indices(base_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            if extra_sample_count > 0:
                extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count, replacement=True).tolist()
                run_indices(extra_indices, opt_contrast, scheduler_contrast, running, trainable_params_contrast)

            print(f"[Full Train | Contrast Ep {ep}] train total={running['total']/Ntr:.4f} | "
                  f"bce={running['bce']/Ntr:.4f} align={running['align']/Ntr:.4f} "
                  f"path={running['path']/Ntr:.4f} label={running['label']/Ntr:.4f} "
                  f"sample={running['sample']/Ntr:.4f}")

    # Reset sample queue before classifier stage
    comb.sample_queue_feats = None
    comb.sample_queue_labels = None

    # ---------------- Stage 2: Classifier training ----------------
    if not cls_on:
        print("[Full Train] M3 disabled (memory_only): skipping classifier training stage.")
        return enc, None, clf_cfg

    comb.loss.use_bce_loss = True  # BCE/Focal 強制開
    comb.loss.use_align_loss = False
    comb.loss.use_label_loss = False
    comb.loss.use_sample_loss = False
    comb.loss.use_path_loss = cfg.use_path_loss
    comb.loss.weight_align = 0.0
    comb.loss.weight_label = 0.0
    comb.loss.weight_sample_contrast = 0.0
    comb.loss.weight_path = cfg.weight_path if cfg.use_path_loss else 0.0

    assert clf is not None
    trainable_params_cls = list(enc.parameters()) + list(clf.parameters())
    param_groups = [
        {"params": enc.parameters(), "lr": cfg.contrast_lr},  # encoder finetune lr matches stage 1
        {"params": clf.parameters(), "lr": cfg.classifier_lr},
    ]
    total_steps_cls = max(1, steps_per_epoch * cfg.classifier_epochs)
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
        running = {"bce": 0.0, "align": 0.0, "path": 0.0, "label": 0.0, "sample": 0.0, "total": 0.0}

        base_indices = torch.randperm(Ntr).tolist()
        run_indices(base_indices, opt, scheduler, running, trainable_params_cls)

        if extra_sample_count > 0:
            extra_indices = torch.multinomial(weight_tensor, num_samples=extra_sample_count, replacement=True).tolist()
            run_indices(extra_indices, opt, scheduler, running, trainable_params_cls)

        print(f"[Full Train | Cls Ep {ep}] train total={running['total']/Ntr:.4f} | "
              f"bce={running['bce']/Ntr:.4f} align={running['align']/Ntr:.4f} "
              f"path={running['path']/Ntr:.4f} label={running['label']/Ntr:.4f} "
              f"sample={running['sample']/Ntr:.4f}")

    return enc, clf, clf_cfg


if __name__ == "__main__":
    main(TrainConfig())
