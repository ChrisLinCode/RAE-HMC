"""Run WOS46985 RAE-HMC experiments across one or more seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from statistics import fmean, pstdev
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import train_rae_hmc as core


CSV_FIELDNAMES = [
    "seed",
    "micro",
    "macro",
    "eta",
    "delta",
    "delta_levels",
    "rho",
    "top_b",
    "top_b_levels",
    "runtime_seconds",
    "runtime_text",
]


@dataclass
class WOSTrainConfig(core.TrainConfig):
    # Dataset
    dataset_csv: str = "dataset/web_of_science/WOS46985.csv"
    hierarchy_json: str = "dataset/web_of_science/WOS46985_hierarchy.json"
    root_label_name: List[str] = field(default_factory=lambda: ["Root"])

    # Split
    test_ratio: float = 0.2
    val_ratio: float = 0.2
    eval_mode: str = "holdout"
    ensure_split_label_coverage: bool = True
    data_fraction: float = 1.0
    quick_mode: bool = False
    quick_fraction: float = 0.1
    seed: int = 41

    # Encoder / runtime
    model_name: str = "bert-base-uncased"
    max_len: int = 512 #512
    batch_size: int = 16 #16 
    cache_tokens_on_gpu: bool = False
    use_bf16_amp: bool = True
    grad_checkpointing: bool = True

    classifier_patience: int = 5

    # WOS-specific validation tuning ranges.
    eta_candidates: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    delta_candidates: List[float] = field(default_factory=lambda: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    top_b_candidates: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15])

    # Output
    workdir: str = "./outputs/wos46985"


def make_wos46985_config() -> WOSTrainConfig:
    return WOSTrainConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAE-HMC on WOS46985 across one or more seeds.")
    parser.add_argument(
        "--retune-from-checkpoint",
        default=None,
        help="Load an existing holdout checkpoint and rerun holdout tuning without retraining.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Path to a saved split_indices.json used with --retune-from-checkpoint. Defaults to the checkpoint directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for the two-stage split.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Run multiple explicit seeds sequentially, e.g. --seeds 41 42 43 44 45.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Inclusive first seed for a sequential multiseed run.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=None,
        help="Inclusive last seed for a sequential multiseed run.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Override directory for checkpoints and outputs.",
    )
    parser.add_argument(
        "--multiseed-subdir",
        default="multiseed_runs",
        help="Subdirectory under --workdir used for per-seed artifacts in multiseed mode.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--max-len", type=int, default=None, help="Override tokenizer max sequence length.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override encoder model name for English WOS abstracts.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Override fraction of the full WOS46985 dataset to use before splitting, e.g. 1.0 or 0.5.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a stratified 10%% subset of WOS46985 for fast parameter checks. "
            "Defaults to outputs/wos46985_quick unless --workdir is provided."
        ),
    )
    parser.add_argument(
        "--quick-fraction",
        type=float,
        default=0.1,
        help="Fraction used by --quick. Default: 0.1, about 4699 samples for WOS46985.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the multiseed result CSV. Default: <workdir>/wos_multiseed.csv.",
    )
    parser.add_argument(
        "--rho-candidates",
        nargs="*",
        type=float,
        default=None,
        help="Override rho search candidates for validation tuning.",
    )
    parser.add_argument(
        "--eta-candidates",
        nargs="*",
        type=float,
        default=None,
        help="Override eta search candidates for validation tuning.",
    )
    parser.add_argument(
        "--delta-candidates",
        nargs="*",
        type=float,
        default=None,
        help="Override delta search candidates for validation tuning.",
    )
    parser.add_argument(
        "--top-b-candidates",
        nargs="*",
        type=int,
        default=None,
        help="Override top-b search candidates for validation tuning.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: WOSTrainConfig, args: argparse.Namespace) -> WOSTrainConfig:
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.workdir is not None:
        cfg.workdir = str(args.workdir)
    if args.quick and args.data_fraction is not None:
        raise ValueError("Use either --quick or --data-fraction, not both.")
    if args.quick:
        cfg.quick_mode = True
        cfg.quick_fraction = normalize_data_fraction(args.quick_fraction)
        cfg.data_fraction = cfg.quick_fraction
        if args.workdir is None:
            cfg.workdir = "./outputs/wos46985_quick"
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.max_len is not None:
        cfg.max_len = int(args.max_len)
    if args.model_name is not None:
        cfg.model_name = str(args.model_name)
    if args.data_fraction is not None:
        cfg.data_fraction = normalize_data_fraction(args.data_fraction)
    if args.rho_candidates is not None:
        cfg.rho_candidates = [float(v) for v in args.rho_candidates]
    if args.eta_candidates is not None:
        cfg.eta_candidates = [float(v) for v in args.eta_candidates]
    if args.delta_candidates is not None:
        cfg.delta_candidates = [float(v) for v in args.delta_candidates]
    if args.top_b_candidates is not None:
        cfg.top_b_candidates = [int(v) for v in args.top_b_candidates]
    return cfg


def resolve_multiseed_args(args: argparse.Namespace) -> Optional[List[int]]:
    explicit_seeds = args.seeds is not None
    range_requested = args.seed_start is not None or args.seed_end is not None

    if args.seed is not None and (explicit_seeds or range_requested):
        raise ValueError("--seed cannot be combined with --seeds or --seed-start/--seed-end.")
    if explicit_seeds and range_requested:
        raise ValueError("Use either --seeds or --seed-start/--seed-end, not both.")
    if explicit_seeds:
        if not args.seeds:
            raise ValueError("--seeds was provided but no seed values were supplied.")
        return list(dict.fromkeys(int(seed) for seed in args.seeds))
    if range_requested:
        if args.seed_start is None or args.seed_end is None:
            raise ValueError("--seed-start and --seed-end must be provided together.")
        if int(args.seed_end) < int(args.seed_start):
            raise ValueError("--seed-end must be >= --seed-start.")
        return list(range(int(args.seed_start), int(args.seed_end) + 1))
    return None


def resolve_retune_args(args: argparse.Namespace) -> Optional[str]:
    checkpoint_path = args.retune_from_checkpoint
    if checkpoint_path is None:
        return None

    forbidden_flags = []
    if args.seed is not None:
        forbidden_flags.append("--seed")
    if args.seeds is not None:
        forbidden_flags.append("--seeds")
    if args.seed_start is not None or args.seed_end is not None:
        forbidden_flags.append("--seed-start/--seed-end")
    if args.quick:
        forbidden_flags.append("--quick")
    if args.data_fraction is not None:
        forbidden_flags.append("--data-fraction")
    if args.quick_fraction != 0.1:
        forbidden_flags.append("--quick-fraction")
    if args.model_name is not None:
        forbidden_flags.append("--model-name")
    if args.max_len is not None:
        forbidden_flags.append("--max-len")
    if args.output_csv is not None:
        forbidden_flags.append("--output-csv")
    if forbidden_flags:
        joined = ", ".join(forbidden_flags)
        raise ValueError(
            "Checkpoint retune mode only reruns validation/test tuning. "
            f"Do not combine it with: {joined}."
        )

    resolved = os.path.abspath(checkpoint_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def default_retune_workdir(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(checkpoint_dir, f"retune_{stamp}")


def load_split_manifest(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Split manifest must be a JSON object: {path}")
    indices = payload.get("indices")
    if not isinstance(indices, dict):
        raise ValueError(f"Split manifest is missing `indices`: {path}")
    return payload


def resolve_split_manifest_path(args: argparse.Namespace, checkpoint_path: str) -> str:
    if args.split_manifest is not None:
        resolved = os.path.abspath(args.split_manifest)
    else:
        resolved = os.path.join(os.path.dirname(checkpoint_path), "split_indices.json")
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Split manifest not found: {resolved}")
    return resolved


def reconstruct_saved_split_indices(
    manifest: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = manifest["indices"]
    selected_idx = np.array(indices["selected"], dtype=int)
    train_global = np.array(indices["train"], dtype=int)
    val_global = np.array(indices["validation"], dtype=int)
    test_global = np.array(indices["test"], dtype=int)

    selected_lookup = {int(global_idx): pos for pos, global_idx in enumerate(selected_idx.tolist())}

    def map_global_to_selected(global_values: np.ndarray, label: str) -> np.ndarray:
        mapped: List[int] = []
        missing: List[int] = []
        for value in global_values.tolist():
            selected_pos = selected_lookup.get(int(value))
            if selected_pos is None:
                missing.append(int(value))
                continue
            mapped.append(int(selected_pos))
        if missing:
            preview = ", ".join(str(v) for v in missing[:5])
            raise ValueError(
                f"Split manifest {label} indices are not contained in selected indices. "
                f"First missing values: {preview}"
            )
        return np.array(mapped, dtype=int)

    train_idx = map_global_to_selected(train_global, "train")
    val_idx = map_global_to_selected(val_global, "validation")
    test_idx = map_global_to_selected(test_global, "test")

    train_pool_idx = np.sort(np.concatenate([train_idx, val_idx])).astype(int)
    pool_lookup = {int(selected_pos): pool_pos for pool_pos, selected_pos in enumerate(train_pool_idx.tolist())}

    def map_selected_to_pool(selected_values: np.ndarray, label: str) -> np.ndarray:
        mapped: List[int] = []
        missing: List[int] = []
        for value in selected_values.tolist():
            pool_pos = pool_lookup.get(int(value))
            if pool_pos is None:
                missing.append(int(value))
                continue
            mapped.append(int(pool_pos))
        if missing:
            preview = ", ".join(str(v) for v in missing[:5])
            raise ValueError(
                f"Split manifest {label} indices are not contained in train_pool indices. "
                f"First missing values: {preview}"
            )
        return np.array(mapped, dtype=int)

    train_rel_idx = map_selected_to_pool(train_idx, "train")
    val_rel_idx = map_selected_to_pool(val_idx, "validation")
    return selected_idx, train_pool_idx, train_rel_idx, val_rel_idx, train_idx, val_idx, test_idx


def format_wos_summary_line(result: Dict[str, float]) -> str:
    rho_val = result.get("rho", None)
    rho_text = f"{float(rho_val):.2f}" if rho_val is not None else "N/A"
    top_b_text = core.format_top_b_display(
        result.get("top_b", 0) if result.get("top_b", None) is not None else 0,
        result.get("top_b_levels", None),
    )
    return (
        f"seed={int(result['seed'])}, micro-F1={float(result['micro']):.4f}, "
        f"macro-F1(all)={float(result['macro_all']):.4f}, eta={float(result['eta']):.2f}, "
        f"delta={core.format_delta_display(result.get('delta', 0.0), result.get('delta_levels', None))}, "
        f"rho={rho_text}, top_b={top_b_text}, time={result.get('runtime_text', 'N/A')}, "
        f"workdir={result['workdir']}"
    )


def json_cell(value: object) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def result_to_csv_row(result: Dict[str, object]) -> Dict[str, str]:
    delta = result.get("delta", "")
    rho = result.get("rho", "")
    top_b = result.get("top_b", "")
    runtime_seconds = result.get("runtime_seconds", "")
    return {
        "seed": str(int(result["seed"])),
        "micro": f"{float(result['micro']):.4f}",
        "macro": f"{float(result['macro_all']):.4f}",
        "eta": f"{float(result['eta']):.4f}",
        "delta": f"{float(delta):.4f}" if delta not in (None, "") else "",
        "delta_levels": json_cell(result.get("delta_levels", None)),
        "rho": f"{float(rho):.4f}" if rho not in (None, "") else "",
        "top_b": str(int(top_b)) if top_b not in (None, "") else "",
        "top_b_levels": json_cell(result.get("top_b_levels", None)),
        "runtime_seconds": f"{float(runtime_seconds):.2f}" if runtime_seconds not in (None, "") else "",
        "runtime_text": str(result.get("runtime_text", "")),
    }


def write_wos_multiseed_csv(csv_path: str, results: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = sorted(results, key=lambda item: int(item["seed"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in rows:
            writer.writerow(result_to_csv_row(result))


def run_multiseed(base_cfg: WOSTrainConfig, seeds: List[int], args: argparse.Namespace) -> None:
    base_workdir = base_cfg.workdir
    os.makedirs(base_workdir, exist_ok=True)
    per_seed_root = os.path.join(base_workdir, args.multiseed_subdir)
    os.makedirs(per_seed_root, exist_ok=True)
    output_csv = args.output_csv or os.path.join(base_workdir, "wos_multiseed.csv")

    results: List[Dict[str, float]] = []
    total = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        seed_workdir = os.path.join(per_seed_root, f"seed{seed}")
        cfg = replace(base_cfg, seed=int(seed), workdir=seed_workdir)
        print(f"\n[{index}/{total}] WOS seed={seed} -> {seed_workdir}")
        result = main(cfg, scenario_name=f"seed{seed}")
        if result is None:
            raise RuntimeError(f"WOS seed {seed} returned None unexpectedly.")

        result["seed"] = int(seed)
        result["workdir"] = seed_workdir
        results.append(result)
        write_wos_multiseed_csv(output_csv, results)

        line = format_wos_summary_line(result)
        print(f"[Multiseed] {line}")
        print(f"[Save] Multiseed CSV updated: {output_csv}")

    micro_vals = [float(item["micro"]) for item in results]
    macro_vals = [float(item["macro_all"]) for item in results]
    micro_mean = fmean(micro_vals)
    macro_mean = fmean(macro_vals)
    micro_std = pstdev(micro_vals) if len(micro_vals) > 1 else 0.0
    macro_std = pstdev(macro_vals) if len(macro_vals) > 1 else 0.0

    write_wos_multiseed_csv(output_csv, results)

    print("\n[Multiseed Aggregate]")
    aggregate_lines = [
        f"seeds={seeds}",
        f"micro-F1={micro_mean:.4f} ± {micro_std:.4f}, n={len(results)}",
        f"macro-F1(all)={macro_mean:.4f} ± {macro_std:.4f}, n={len(results)}",
    ]
    for line in aggregate_lines:
        print(f"  - {line}")


def normalize_data_fraction(value: float) -> float:
    frac = float(value)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"data_fraction must be in (0, 1], got {value}")
    return frac


def select_subset_indices(
    Y_all: np.ndarray,
    cfg: WOSTrainConfig,
) -> np.ndarray:
    total = int(len(Y_all))
    frac = normalize_data_fraction(getattr(cfg, "data_fraction", 1.0))
    if frac >= 1.0:
        return np.arange(total, dtype=int)

    keep_count = max(1, min(total, int(np.floor(total * frac + 0.5))))
    ensure_coverage = bool(getattr(cfg, "ensure_split_label_coverage", True))
    if Y_all.ndim == 2 and keep_count < int(Y_all.shape[1]):
        ensure_coverage = False

    _, keep_idx = core.iterative_stratified_split(
        Y_all,
        test_size=keep_count / float(total),
        seed=cfg.seed + 17,
        ensure_test_label_coverage=ensure_coverage,
    )
    keep_idx = np.sort(keep_idx.astype(int))
    return keep_idx


def save_split_manifest(
    cfg: WOSTrainConfig,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    total_samples: int,
) -> str:
    manifest = {
        "dataset_csv": cfg.dataset_csv,
        "hierarchy_json": cfg.hierarchy_json,
        "seed": int(cfg.seed),
        "test_ratio": float(cfg.test_ratio),
        "val_ratio": float(cfg.val_ratio),
        "data_fraction": float(cfg.data_fraction),
        "quick_mode": bool(getattr(cfg, "quick_mode", False)),
        "quick_fraction": (
            float(getattr(cfg, "quick_fraction"))
            if getattr(cfg, "quick_mode", False)
            else None
        ),
        "counts": {
            "selected": int(len(subset_idx)),
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
            "total": int(total_samples),
        },
        "indices": {
            "selected": subset_idx.tolist(),
            "train": train_idx.tolist(),
            "validation": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }
    path = os.path.join(cfg.workdir, "split_indices.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def append_wos_result_txt(cfg: WOSTrainConfig, lines: List[str]) -> str:
    path = os.path.join(cfg.workdir, "wos_result.txt")
    os.makedirs(cfg.workdir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n\n")
        f.flush()
    return path


def retune_from_checkpoint(
    cfg: WOSTrainConfig,
    checkpoint_path: str,
    split_manifest_path: str,
    scenario_name: Optional[str] = None,
) -> Dict[str, object]:
    os.makedirs(cfg.workdir, exist_ok=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    print(
        f"[AMP] bf16_amp={'on' if bool(getattr(cfg, 'use_bf16_amp', False)) and device.type == 'cuda' else 'off'} "
        f"| grad_checkpointing={'on' if bool(getattr(cfg, 'grad_checkpointing', False)) else 'off'}"
    )

    manifest = load_split_manifest(split_manifest_path)
    manifest_seed = int(manifest.get("seed", cfg.seed))
    manifest_dataset_csv = str(manifest.get("dataset_csv", cfg.dataset_csv))
    manifest_hierarchy_json = str(manifest.get("hierarchy_json", cfg.hierarchy_json))
    cfg.seed = manifest_seed
    cfg.dataset_csv = manifest_dataset_csv
    cfg.hierarchy_json = manifest_hierarchy_json
    core.set_seed(cfg.seed)
    local_split_manifest_path = os.path.join(cfg.workdir, "split_indices.json")
    with open(local_split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    root_names = core.parse_root_label_names(getattr(cfg, "root_label_name", "Root"))
    if bool(getattr(cfg, "exclude_root_label", False)):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as f:
            hjson = json.load(f)
        hjson = core.promote_named_roots(hjson, root_names)
        hd = core.parse_label_hierarchy(hjson)
    else:
        hd = core.load_hierarchy_from_file(cfg.hierarchy_json)
    L = hd.num_labels
    print(f"[Hierarchy] num_labels={L}, level_sizes={hd.level_sizes}")
    level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
    label_levels = [level_lookup.get(i, 1) for i in range(L)]

    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [core.parse_label_cell(s) for s in df_all[cfg.labels_col].tolist()]
    if bool(getattr(cfg, "exclude_root_label", False)):
        all_label_lists = [core.strip_root_label(labs, root_names) for labs in all_label_lists]
    Y_all = np.array(core.build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True))

    (
        subset_idx,
        train_pool_idx,
        train_rel_idx,
        val_rel_idx,
        train_idx,
        val_idx,
        test_idx,
    ) = reconstruct_saved_split_indices(manifest)
    df_selected = df_all.iloc[subset_idx].reset_index(drop=True)
    Y_selected = Y_all[subset_idx]
    print(
        f"[Data] Selected {len(subset_idx)} / {len(df_all)} samples "
        f"(saved split manifest)"
    )
    print(
        f"[Data] Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)} "
        f"(restored from {split_manifest_path})"
    )

    df_train_pool = df_selected.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_selected.iloc[test_idx].reset_index(drop=True)
    train_pool_texts = df_train_pool[cfg.text_col].astype(str).tolist()
    test_texts = df_test[cfg.text_col].astype(str).tolist()

    Y_tr_full = torch.tensor(Y_selected[train_pool_idx], dtype=torch.float32)
    Y_te = torch.tensor(Y_selected[test_idx], dtype=torch.float32)

    print("[Stage] Initializing shared encoder...")
    enc = core.SharedEncoder(
        core.EncoderConfig(
            model_name=cfg.model_name,
            max_length=cfg.max_len,
            pooling=cfg.encoder_pooling,
            normalize=True,
            device=device_str,
        )
    )
    print("[Stage] Tokenizing train/test/label texts...")
    label_descs = core.build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
    train_tokens = core.tokenize_texts(enc.tokenizer, train_pool_texts, cfg.max_len)
    test_tokens = core.tokenize_texts(enc.tokenizer, test_texts, cfg.max_len)
    label_tokens = core.tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
    del enc

    level_slices = core.make_level_slices(hd.levels)
    hierarchy_obj = core.Hierarchy(num_labels=L, ancestors=hd.ancestors)
    mem_cfg = core.MemoryConfig(
        backend="faiss_ip",
        top_b=cfg.top_b,
        tau_mem=cfg.tau_mem,
        rho=cfg.rho,
        device=device_str,
    )

    train_test_start = time.perf_counter()
    print(f"[Retune] Loading checkpoint: {checkpoint_path}")
    enc, clf, checkpoint = core.load_model_from_checkpoint_for_test(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        device_str=device_str,
    )

    eval_train_tokens = core.subset_tokens(train_tokens, train_rel_idx)
    eval_train_labels = Y_tr_full.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))
    val_tokens_for_tuning = core.subset_tokens(train_tokens, val_rel_idx)
    val_labels_for_tuning = Y_tr_full.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))

    print(
        "[Retune] Candidate ranges: "
        f"rho={cfg.rho_candidates}, eta={cfg.eta_candidates}, "
        f"delta={cfg.delta_candidates}, top_b={cfg.top_b_candidates}"
    )
    retune_result = core.tune_validation_strategy(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        label_tokens=label_tokens,
        train_tokens=eval_train_tokens,
        Y_tr=eval_train_labels,
        val_tokens=val_tokens_for_tuning,
        Y_va=val_labels_for_tuning,
        enc=enc,
        clf=clf,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg,
    )
    print(
        f"[Retune] val micro-F1={retune_result['micro']:.4f}, "
        f"macro-F1={retune_result['macro']:.4f} {retune_result['tuning_info']}"
    )

    eta_final = float(retune_result["eta"])
    delta_final = float(retune_result["delta"])
    delta_levels_final = retune_result.get("delta_levels", None)
    rho_final = float(retune_result["rho"])
    top_b_final = int(retune_result["top_b"])
    top_b_levels_final = retune_result.get("top_b_levels", None)
    mem_cfg_eval = replace(mem_cfg, rho=rho_final, top_b=top_b_final)

    test_result = core.evaluate_model_on_test_split(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        label_levels=label_levels,
        label_tokens=label_tokens,
        train_tokens_for_memory=eval_train_tokens,
        Y_train_for_memory=eval_train_labels,
        test_tokens=test_tokens,
        Y_te=Y_te,
        enc=enc,
        clf=clf,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg_eval,
        eta_final=eta_final,
        delta_final=delta_final,
        top_b_final=top_b_final,
        top_b_levels_final=top_b_levels_final,
        delta_levels_final=delta_levels_final,
    )

    mem = test_result["mem"]
    y_true_te = test_result["y_true_te"]
    y_pred_te = test_result["y_pred_te"]
    micro = test_result["micro"]
    macro_all = test_result["macro_all"]
    if bool(getattr(cfg, "print_per_label_metrics", True)):
        print("[TEST] Per-label metrics:")
        core.per_label_report(y_true_te, y_pred_te, hd.id2label)
    print(
        f"[Final Tuning] Using eta={eta_final:.2f}, delta={delta_final:.2f}, "
        f"rho={rho_final:.2f}, top_b={core.format_top_b_display(top_b_final, top_b_levels_final)} "
        "derived from checkpoint retune."
    )
    if core.normalize_delta_mode(cfg) == "level" and delta_levels_final:
        levels_sorted = sorted(delta_levels_final.items(), key=lambda x: x[0])
        levels_str = ", ".join([f"L{lvl}={val:.2f}" for lvl, val in levels_sorted])
        print(f"[Final Tuning] Per-level delta: {levels_str}")
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")

    export_path = os.path.join(cfg.workdir, "test_result.xlsx")
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
        rows.append(
            {
                "text": texts[i],
                "label": "; ".join(true_labels),
                "pred_label": "; ".join(pred_labels),
                "false_positive": "; ".join(fp),
                "false_negative": "; ".join(fn),
            }
        )
    df_out = pd.DataFrame(rows)
    try:
        df_out.to_excel(export_path, index=False)
        print(f"[Save] Test predictions saved to {export_path}")
        export_saved_path = export_path
    except Exception as exc:
        fallback = os.path.splitext(export_path)[0] + ".csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[Warn] Failed to write xlsx: {exc}. Saved CSV instead: {fallback}")
        export_saved_path = fallback

    if mem is not None:
        mem.save(os.path.join(cfg.workdir, "memory_store"))
    with open(os.path.join(cfg.workdir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": hd.label2id, "id2label": {int(k): v for k, v in hd.id2label.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(cfg.workdir, "ancestors.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in hd.ancestors.items()}, f, ensure_ascii=False, indent=2)

    runtime_seconds = time.perf_counter() - train_test_start
    runtime_text = core.format_elapsed_time(runtime_seconds)
    print(f"[Done] Artifacts saved to {cfg.workdir}")
    print(f"[Time] Retune+test elapsed: {runtime_text} ({runtime_seconds:.2f}s)")

    result_txt_path = append_wos_result_txt(
        cfg,
        [
            f"[{datetime.now().isoformat(timespec='seconds')}] Checkpoint Retune",
            f"seed={cfg.seed}, batch_size={cfg.batch_size}, max_len={cfg.max_len}, model_name={cfg.model_name}",
            f"checkpoint={checkpoint_path}",
            f"split_manifest_source={split_manifest_path}",
            f"split_manifest={local_split_manifest_path}",
            f"split_counts: selected={len(subset_idx)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            (
                "candidates: "
                f"rho={cfg.rho_candidates}, eta={cfg.eta_candidates}, "
                f"delta={cfg.delta_candidates}, top_b={cfg.top_b_candidates}"
            ),
            (
                "checkpoint_defaults: "
                f"rho={checkpoint.get('rho')}, eta={checkpoint.get('eta')}, "
                f"delta={checkpoint.get('delta')}, top_b={checkpoint.get('top_b')}"
            ),
            (
                f"val_micro_f1={retune_result['micro']:.4f}, "
                f"val_macro_f1={retune_result['macro']:.4f}, "
                f"eta={eta_final:.4f}, delta={delta_final:.4f}, rho={rho_final:.4f}, "
                f"top_b={core.format_top_b_display(top_b_final, top_b_levels_final)}"
            ),
            f"test_predictions={export_saved_path}",
            f"test_micro_f1={micro:.4f}",
            f"test_macro_f1_all={macro_all:.4f}",
            f"elapsed={runtime_text} ({runtime_seconds:.2f}s)",
        ],
    )
    print(f"[Save] Retune summary appended to {result_txt_path}")

    scenario = scenario_name if scenario_name is not None else os.path.basename(cfg.workdir.rstrip(os.sep))
    return {
        "scenario": scenario,
        "eta": eta_final,
        "delta": delta_final,
        "delta_levels": delta_levels_final,
        "rho": (rho_final if cfg.use_memory else None),
        "top_b": (top_b_final if cfg.use_memory else None),
        "top_b_levels": (top_b_levels_final if cfg.use_memory else None),
        "micro": micro,
        "macro_all": macro_all,
        "runtime_seconds": runtime_seconds,
        "runtime_text": runtime_text,
        "holdout_checkpoint": checkpoint_path,
        "full_checkpoint": None,
        "test_predictions": export_saved_path,
        "use_memory": bool(cfg.use_memory),
        "use_global_branch": bool(cfg.use_global_branch),
        "use_local_branch": bool(cfg.use_local_branch),
    }


def main(
    cfg: WOSTrainConfig,
    summary: Optional[List[Dict[str, float]]] = None,
    scenario_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    os.makedirs(cfg.workdir, exist_ok=True)
    core.set_seed(cfg.seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    print(
        f"[AMP] bf16_amp={'on' if bool(getattr(cfg, 'use_bf16_amp', False)) and device.type == 'cuda' else 'off'} "
        f"| grad_checkpointing={'on' if bool(getattr(cfg, 'grad_checkpointing', False)) else 'off'}"
    )

    root_names = core.parse_root_label_names(getattr(cfg, "root_label_name", "Root"))
    if bool(getattr(cfg, "exclude_root_label", False)):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as f:
            hjson = json.load(f)
        hjson = core.promote_named_roots(hjson, root_names)
        hd = core.parse_label_hierarchy(hjson)
    else:
        hd = core.load_hierarchy_from_file(cfg.hierarchy_json)
    L = hd.num_labels
    print(f"[Hierarchy] num_labels={L}, level_sizes={hd.level_sizes}")
    level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
    label_levels = [level_lookup.get(i, 1) for i in range(L)]

    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [core.parse_label_cell(s) for s in df_all[cfg.labels_col].tolist()]
    if bool(getattr(cfg, "exclude_root_label", False)):
        all_label_lists = [core.strip_root_label(labs, root_names) for labs in all_label_lists]
    Y_all = np.array(core.build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True))

    subset_idx = select_subset_indices(Y_all, cfg)
    df_selected = df_all.iloc[subset_idx].reset_index(drop=True)
    Y_selected = Y_all[subset_idx]
    print(
        f"[Data] Selected {len(subset_idx)} / {len(df_all)} samples "
        f"(data_fraction={cfg.data_fraction:.4f})"
    )

    train_pool_idx, train_rel_idx, val_rel_idx, train_idx, val_idx, test_idx = core.build_two_stage_split_indices(
        Y_selected, cfg
    )
    split_manifest_path = save_split_manifest(
        cfg,
        subset_idx=subset_idx,
        train_idx=subset_idx[train_idx],
        val_idx=subset_idx[val_idx],
        test_idx=subset_idx[test_idx],
        total_samples=len(df_all),
    )
    print(
        f"[Data] Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)} "
        f"(two-stage stratified split, seed={cfg.seed})"
    )
    print(f"[Save] Split indices saved to {split_manifest_path}")

    df_train_pool = df_selected.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_selected.iloc[test_idx].reset_index(drop=True)
    train_pool_texts = df_train_pool[cfg.text_col].astype(str).tolist()
    test_texts = df_test[cfg.text_col].astype(str).tolist()

    Y_tr_full = torch.tensor(Y_selected[train_pool_idx], dtype=torch.float32)
    Y_te = torch.tensor(Y_selected[test_idx], dtype=torch.float32)

    print("[Stage] Initializing shared encoder...")
    enc = core.SharedEncoder(
        core.EncoderConfig(
            model_name=cfg.model_name,
            max_length=cfg.max_len,
            pooling=cfg.encoder_pooling,
            normalize=True,
            device=device_str,
        )
    )
    print("[Stage] Tokenizing train/test/label texts...")
    label_descs = core.build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
    train_tokens = core.tokenize_texts(enc.tokenizer, train_pool_texts, cfg.max_len)
    test_tokens = core.tokenize_texts(enc.tokenizer, test_texts, cfg.max_len)
    label_tokens = core.tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
    del enc

    level_slices = core.make_level_slices(hd.levels)
    edges_pc = [(int(p), int(c)) for (p, c) in hd.edges_parent_child]
    hierarchy_obj = core.Hierarchy(num_labels=L, ancestors=hd.ancestors)

    memory_backend = "faiss_ip"
    print(f"[Memory] backend={memory_backend}")
    mem_cfg = core.MemoryConfig(
        backend=memory_backend,
        top_b=cfg.top_b,
        tau_mem=cfg.tau_mem,
        rho=cfg.rho,
        device=device_str,
    )

    train_test_start = time.perf_counter()
    fold_result = core.train_single_fold(
        fold_name="holdout",
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        edges_pc=edges_pc,
        label_tokens=label_tokens,
        train_tokens_full=train_tokens,
        Y_train_full=Y_tr_full,
        train_indices=train_rel_idx,
        val_indices=val_rel_idx,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg,
    )

    val_metric = core.get_val_metric_name(cfg)
    metric_val = float(fold_result.get("best_val_score", -1.0))
    micro_val = float(fold_result.get("best_val_micro", -1.0))
    macro_val = float(fold_result.get("best_val_macro", -1.0))
    rho_info = fold_result.get("last_tuned_rho", None)
    eta_info = fold_result.get("last_tuned_eta", None)
    delta_info = fold_result.get("last_tuned_delta", None)
    delta_levels_info = fold_result.get("last_tuned_delta_levels", None)
    top_b_info = fold_result.get("last_tuned_top_b", None)
    top_b_levels_info = fold_result.get("last_tuned_top_b_levels", None)
    delta_text = core.format_delta_display(delta_info, delta_levels_info) if delta_info is not None else "N/A"
    top_b_print = (
        f", top_b={core.format_top_b_display(top_b_info if top_b_info is not None else cfg.top_b, top_b_levels_info)}"
        if (top_b_info is not None or top_b_levels_info is not None)
        else ""
    )
    print("\n[Holdout] Validation summary:")
    if bool(fold_result.get("use_memory", True)) and rho_info is not None:
        holdout_summary = (
            f"holdout: best val {val_metric}-F1={metric_val:.4f}, micro-F1={micro_val:.4f}, "
            f"macro-F1={macro_val:.4f}, rho={rho_info:.2f}, eta={eta_info:.2f}, "
            f"delta={delta_text}{top_b_print}, checkpoint={fold_result['best_path']}"
        )
        print(f"  - {holdout_summary}")
    else:
        holdout_summary = (
            f"holdout: best val {val_metric}-F1={metric_val:.4f}, micro-F1={micro_val:.4f}, "
            f"macro-F1={macro_val:.4f}, checkpoint={fold_result['best_path']}"
        )
        print(f"  - {holdout_summary}")

    result_txt_path = append_wos_result_txt(
        cfg,
        [
            f"[{datetime.now().isoformat(timespec='seconds')}] Holdout Validation",
            f"data_fraction={cfg.data_fraction:.4f}, seed={cfg.seed}, batch_size={cfg.batch_size}, max_len={cfg.max_len}, model_name={cfg.model_name}",
            f"split_counts: selected={len(subset_idx)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            holdout_summary,
            f"split_manifest={split_manifest_path}",
        ],
    )
    print(f"[Save] Holdout summary appended to {result_txt_path}")

    eta_final = fold_result["last_tuned_eta"] if fold_result.get("last_tuned_eta") is not None else cfg.eta
    delta_final = fold_result["last_tuned_delta"] if fold_result.get("last_tuned_delta") is not None else cfg.delta
    delta_levels_final = fold_result.get("last_tuned_delta_levels", None)
    rho_final = fold_result["last_tuned_rho"] if fold_result.get("last_tuned_rho") is not None else cfg.rho
    top_b_final = fold_result["last_tuned_top_b"] if fold_result.get("last_tuned_top_b") is not None else cfg.top_b
    top_b_levels_final = fold_result.get("last_tuned_top_b_levels", None)
    selected_checkpoint_path = fold_result.get("best_path")
    final_tuning_source = "holdout validation"

    if not selected_checkpoint_path or not os.path.exists(selected_checkpoint_path):
        raise RuntimeError("Holdout best checkpoint is missing; cannot evaluate holdout-best model on test.")
    print(f"\n[Test Mode] Using holdout-best checkpoint directly: {selected_checkpoint_path}")
    enc, clf, checkpoint = core.load_model_from_checkpoint_for_test(
        cfg=cfg,
        checkpoint_path=selected_checkpoint_path,
        device=device,
        device_str=device_str,
    )
    eta_final = float(checkpoint.get("eta", eta_final))
    delta_final = float(checkpoint.get("delta", delta_final))
    delta_levels_final = checkpoint.get("delta_levels", delta_levels_final)
    rho_final = float(checkpoint.get("rho", rho_final))
    top_b_final = int(checkpoint.get("top_b", top_b_final))
    top_b_levels_final = checkpoint.get("top_b_levels", top_b_levels_final)
    eval_train_tokens = core.subset_tokens(train_tokens, train_rel_idx)
    eval_train_labels = Y_tr_full.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))

    retrieval_protocol = core.normalize_retrieval_protocol(cfg)
    if bool(cfg.use_memory) and retrieval_protocol == "post_hoc":
        val_tokens_for_tuning = core.subset_tokens(train_tokens, val_rel_idx)
        val_labels_for_tuning = Y_tr_full.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))
        posthoc_result = core.tune_validation_strategy(
            cfg=cfg,
            hd=hd,
            hierarchy_obj=hierarchy_obj,
            level_slices=level_slices,
            label_levels=label_levels,
            label_tokens=label_tokens,
            train_tokens=eval_train_tokens,
            Y_tr=eval_train_labels,
            val_tokens=val_tokens_for_tuning,
            Y_va=val_labels_for_tuning,
            enc=enc,
            clf=clf,
            device=device,
            device_str=device_str,
            mem_cfg=mem_cfg,
        )
        eta_final = float(posthoc_result["eta"])
        delta_final = float(posthoc_result["delta"])
        delta_levels_final = posthoc_result.get("delta_levels", None)
        rho_final = float(posthoc_result["rho"])
        top_b_final = int(posthoc_result["top_b"])
        top_b_levels_final = posthoc_result.get("top_b_levels", None)
        final_tuning_source = "post-hoc holdout validation"
        print(
            f"[Post-hoc Retrieval Tuning] val micro-F1={posthoc_result['micro']:.4f}, "
            f"macro-F1={posthoc_result['macro']:.4f} {posthoc_result['tuning_info']}"
        )
    elif bool(cfg.use_memory):
        final_tuning_source = "per-epoch holdout validation"
    mem_cfg_eval = replace(mem_cfg, rho=rho_final, top_b=top_b_final)

    test_result = core.evaluate_model_on_test_split(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        label_levels=label_levels,
        label_tokens=label_tokens,
        train_tokens_for_memory=eval_train_tokens,
        Y_train_for_memory=eval_train_labels,
        test_tokens=test_tokens,
        Y_te=Y_te,
        enc=enc,
        clf=clf,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg_eval,
        eta_final=eta_final,
        delta_final=delta_final,
        top_b_final=top_b_final,
        top_b_levels_final=top_b_levels_final,
        delta_levels_final=delta_levels_final,
    )

    mem = test_result["mem"]
    y_true_te = test_result["y_true_te"]
    y_pred_te = test_result["y_pred_te"]
    micro = test_result["micro"]
    macro_all = test_result["macro_all"]
    if bool(getattr(cfg, "print_per_label_metrics", True)):
        print("[TEST] Per-label metrics:")
        core.per_label_report(y_true_te, y_pred_te, hd.id2label)
    print(
        f"[Final Tuning] Using eta={eta_final:.2f}, delta={delta_final:.2f}, "
        f"rho={rho_final:.2f}, top_b={core.format_top_b_display(top_b_final, top_b_levels_final)} "
        f"derived from {final_tuning_source}."
    )
    if core.normalize_delta_mode(cfg) == "level" and delta_levels_final:
        levels_sorted = sorted(delta_levels_final.items(), key=lambda x: x[0])
        levels_str = ", ".join([f"L{lvl}={val:.2f}" for lvl, val in levels_sorted])
        print(f"[Final Tuning] Per-level delta: {levels_str}")
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")

    export_path = os.path.join(cfg.workdir, "test_result.xlsx")
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
        rows.append(
            {
                "text": texts[i],
                "label": "; ".join(true_labels),
                "pred_label": "; ".join(pred_labels),
                "false_positive": "; ".join(fp),
                "false_negative": "; ".join(fn),
            }
        )
    df_out = pd.DataFrame(rows)
    try:
        df_out.to_excel(export_path, index=False)
        print(f"[Save] Test predictions saved to {export_path}")
        export_saved_path = export_path
    except Exception as exc:
        fallback = os.path.splitext(export_path)[0] + ".csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[Warn] Failed to write xlsx: {exc}. Saved CSV instead: {fallback}")
        export_saved_path = fallback

    if mem is not None:
        mem.save(os.path.join(cfg.workdir, "memory_store"))
    full_ckpt_path = None
    print(f"[Save] Reused holdout-best checkpoint for test: {selected_checkpoint_path}")
    with open(os.path.join(cfg.workdir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": hd.label2id, "id2label": {int(k): v for k, v in hd.id2label.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(cfg.workdir, "ancestors.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in hd.ancestors.items()}, f, ensure_ascii=False, indent=2)
    print(f"[Done] Artifacts saved to {cfg.workdir}")
    runtime_seconds = time.perf_counter() - train_test_start
    runtime_text = core.format_elapsed_time(runtime_seconds)
    print(f"[Time] Train+test elapsed: {runtime_text} ({runtime_seconds:.2f}s)")

    final_result_txt_path = append_wos_result_txt(
        cfg,
        [
            f"[{datetime.now().isoformat(timespec='seconds')}] Final Test Evaluation",
            f"data_fraction={cfg.data_fraction:.4f}, seed={cfg.seed}, batch_size={cfg.batch_size}, max_len={cfg.max_len}, model_name={cfg.model_name}",
            f"split_counts: selected={len(subset_idx)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            f"holdout_checkpoint={fold_result['best_path']}",
            f"split_manifest={split_manifest_path}",
            f"test_predictions={export_saved_path}",
            f"test_micro_f1={micro:.4f}",
            f"test_macro_f1_all={macro_all:.4f}",
            f"elapsed={runtime_text} ({runtime_seconds:.2f}s)",
            f"eta={eta_final:.4f}, delta={delta_final:.4f}, rho={rho_final:.4f}, "
            f"top_b={core.format_top_b_display(top_b_final, top_b_levels_final)}",
        ]
        + ([f"full_checkpoint={full_ckpt_path}"] if full_ckpt_path is not None else []),
    )
    print(f"[Save] Final evaluation appended to {final_result_txt_path}")

    scenario = scenario_name if scenario_name is not None else os.path.basename(cfg.workdir.rstrip(os.sep))
    return {
        "scenario": scenario,
        "eta": eta_final,
        "delta": delta_final,
        "delta_levels": delta_levels_final,
        "rho": (rho_final if cfg.use_memory else None),
        "top_b": (top_b_final if cfg.use_memory else None),
        "top_b_levels": (top_b_levels_final if cfg.use_memory else None),
        "micro": micro,
        "macro_all": macro_all,
        "runtime_seconds": runtime_seconds,
        "runtime_text": runtime_text,
        "holdout_checkpoint": selected_checkpoint_path,
        "full_checkpoint": full_ckpt_path,
        "test_predictions": export_saved_path,
        "use_memory": bool(cfg.use_memory),
        "use_global_branch": bool(cfg.use_global_branch),
        "use_local_branch": bool(cfg.use_local_branch),
    }


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = resolve_retune_args(args)
    seeds = resolve_multiseed_args(args)
    cfg = make_wos46985_config()
    cfg = apply_cli_overrides(cfg, args)
    if checkpoint_path is not None:
        split_manifest_path = resolve_split_manifest_path(args, checkpoint_path)
        if args.workdir is None:
            cfg.workdir = default_retune_workdir(checkpoint_path)
        retune_from_checkpoint(cfg, checkpoint_path=checkpoint_path, split_manifest_path=split_manifest_path)
    elif seeds is None:
        main(cfg)
    else:
        run_multiseed(cfg, seeds, args)
