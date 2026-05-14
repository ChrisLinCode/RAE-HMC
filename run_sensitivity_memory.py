#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from build_hierarchy_utils import build_multi_hot_Y, make_level_slices, parse_label_hierarchy, load_hierarchy_from_file
from inference import Hierarchy, InferenceConfig, InferenceEngine
from memory import MemoryConfig
from train_rae_hmc import (
    TrainConfig,
    build_encoder_config,
    build_label_descriptions,
    build_memory_store,
    build_two_stage_split_indices,
    encode_with_encoder,
    load_model_from_checkpoint_for_test,
    macro_f1,
    main as train_main,
    micro_f1,
    parse_label_cell,
    parse_root_label_names,
    predict_with_strategy,
    prepare_memory_inputs,
    promote_named_roots,
    set_seed,
    strip_root_label,
    subset_tokens,
    tune_validation_strategy,
    tokenize_texts,
)
from encoder import SharedEncoder


CSV_FIELDNAMES = ["seed", "param_name", "param_value", "micro", "macro"]
DEFAULT_WORKDIR = "outputs/sensitivity_memory"
DEFAULT_OUTPUT_CSV = "outputs/sensitivity_memory/sensitivity_memory_sum.csv"
DEFAULT_CHECKPOINT_ROOT = "outputs/ablation_cl/ablation_cl_runs"
DEFAULT_CHECKPOINT_SCENARIO = "raehmc_cl"
DEFAULT_POSTHOC_SUMMARY_ROOT = ""
DEFAULT_POSTHOC_SUMMARY_SCENARIO = ""
DEFAULT_BASELINE_SUMMARY_DIR = "outputs/sensitivity_memory/posthoc_baseline_params"
DEFAULT_RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_TOP_B_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 20, 30]
DEFAULT_ETA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc sensitivity analysis for memory/fusion parameters. "
            "By default, uses the final RAE-HMC checkpoints "
            "(L_SS+L_HNM, direct-sum fusion) from CL ablation, tunes post-hoc "
            "retrieval parameters on holdout validation, and sweeps rho/top_b/eta without retraining."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--seed-end", type=int, default=70)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "External checkpoint root with seed directories. "
            "Expected layout: <root>/seed<seed>/<checkpoint-scenario>/best_model_holdout.pt. "
            "Set to an empty string to train/use checkpoints under --workdir instead."
        ),
    )
    parser.add_argument(
        "--checkpoint-scenario",
        default=DEFAULT_CHECKPOINT_SCENARIO,
        help="Scenario directory under each seed folder when --checkpoint-root is used.",
    )
    parser.add_argument(
        "--posthoc-summary-root",
        default=DEFAULT_POSTHOC_SUMMARY_ROOT,
        help=(
            "Root with post-hoc summary directories. "
            "Expected layout: <root>/seed<seed>/<posthoc-summary-scenario>/posthoc_eval_summary.json. "
            "Set to an empty string to tune baseline parameters from holdout validation."
        ),
    )
    parser.add_argument(
        "--posthoc-summary-scenario",
        default=DEFAULT_POSTHOC_SUMMARY_SCENARIO,
        help="Scenario directory containing posthoc_eval_summary.json.",
    )
    parser.add_argument(
        "--baseline-summary-dir",
        default=DEFAULT_BASELINE_SUMMARY_DIR,
        help=(
            "Directory for saving per-seed tuned baseline parameter summaries. "
            "Each seed is saved to <dir>/seed<seed>/posthoc_baseline_summary.json. "
            "Set to an empty string to disable saving."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing checkpoints and CSV rows when present.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Recompute existing CSV rows while still reusing checkpoints with --resume.")
    parser.add_argument("--force-train", action="store_true", help="Train even if a checkpoint already exists.")
    parser.add_argument("--rho-values", type=float, nargs="+", default=DEFAULT_RHO_VALUES)
    parser.add_argument("--top-b-values", type=int, nargs="+", default=DEFAULT_TOP_B_VALUES)
    parser.add_argument("--eta-values", type=float, nargs="+", default=DEFAULT_ETA_VALUES)
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return sorted(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def format_float(value: float) -> str:
    return f"{float(value):.6g}"


def row_key(seed: int, param_name: str, param_value: object) -> Tuple[int, str, str]:
    return int(seed), str(param_name), str(param_value)


def load_existing_rows(csv_path: str) -> Dict[Tuple[int, str, str], Dict[str, str]]:
    rows: Dict[Tuple[int, str, str], Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw:
                continue
            seed_text = (raw.get("seed") or "").strip()
            param_name = (raw.get("param_name") or "").strip()
            param_value = (raw.get("param_value") or "").strip()
            if not seed_text or not param_name or not param_value:
                continue
            rows[row_key(int(seed_text), param_name, param_value)] = {
                field: raw.get(field, "") for field in CSV_FIELDNAMES
            }
    return rows


def write_rows(csv_path: str, rows: Dict[Tuple[int, str, str], Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (int(r["seed"]), r["param_name"], float(r["param_value"])))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in ordered:
            row = dict(row)
            for fieldname in ("micro", "macro"):
                if row.get(fieldname, "").strip():
                    row[fieldname] = f"{float(row[fieldname]):.4f}"
            writer.writerow(row)


def checkpoint_path_for(workdir: str) -> str:
    return os.path.join(workdir, "best_model_holdout.pt")


def external_checkpoint_path_for(checkpoint_root: str, seed: int, checkpoint_scenario: str) -> str:
    return os.path.join(
        checkpoint_root,
        f"seed{int(seed)}",
        checkpoint_scenario,
        "best_model_holdout.pt",
    )


def resolve_checkpoint_path(
    cfg: TrainConfig,
    checkpoint_root: str,
    checkpoint_scenario: str,
    resume: bool,
    force_train: bool,
) -> str:
    checkpoint_root = str(checkpoint_root or "").strip()
    checkpoint_scenario = str(checkpoint_scenario or "").strip()
    if checkpoint_root:
        if not checkpoint_scenario:
            raise ValueError("--checkpoint-scenario is required when --checkpoint-root is set.")
        checkpoint_path = external_checkpoint_path_for(checkpoint_root, int(cfg.seed), checkpoint_scenario)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"External checkpoint not found: {checkpoint_path}")
        print(f"[Seed {cfg.seed}] Using external checkpoint: {checkpoint_path}")
        return checkpoint_path
    return ensure_checkpoint(cfg, resume=resume, force_train=force_train)


def external_posthoc_summary_path_for(summary_root: str, seed: int, summary_scenario: str) -> str:
    return os.path.join(
        summary_root,
        f"seed{int(seed)}",
        summary_scenario,
        "posthoc_eval_summary.json",
    )


def load_posthoc_baseline_params(summary_root: str, seed: int, summary_scenario: str) -> Dict[str, object]:
    summary_root = str(summary_root or "").strip()
    summary_scenario = str(summary_scenario or "").strip()
    if not summary_root:
        print(f"[Seed {seed}] No post-hoc summary root set; tuning baseline parameters on holdout validation.")
        return {}
    if not summary_scenario:
        raise ValueError("--posthoc-summary-scenario is required when --posthoc-summary-root is set.")

    summary_path = external_posthoc_summary_path_for(summary_root, int(seed), summary_scenario)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Post-hoc summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(f"[Seed {seed}] Using post-hoc baseline parameters: {summary_path}")
    return payload


def baseline_params_from_holdout(
    *,
    cfg: TrainConfig,
    hd,
    hierarchy_obj: Hierarchy,
    label_levels: List[int],
    label_tokens: Dict[str, torch.Tensor],
    train_tokens: Dict[str, torch.Tensor],
    Y_tr: torch.Tensor,
    val_tokens: Dict[str, torch.Tensor],
    Y_va: torch.Tensor,
    enc,
    clf,
    device: torch.device,
    device_str: str,
) -> Dict[str, object]:
    mem_cfg = MemoryConfig(
        backend="faiss_ip",
        top_b=cfg.top_b,
        tau_mem=cfg.tau_mem,
        rho=cfg.rho,
        device=device_str,
    )
    result = tune_validation_strategy(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=make_level_slices(hd.levels),
        label_levels=label_levels,
        label_tokens=label_tokens,
        train_tokens=train_tokens,
        Y_tr=Y_tr,
        val_tokens=val_tokens,
        Y_va=Y_va,
        enc=enc,
        clf=clf,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg,
    )
    print(
        f"[Seed {cfg.seed}] Holdout post-hoc baseline: "
        f"micro={float(result['micro']):.4f}, macro={float(result['macro']):.4f} "
        f"{result['tuning_info']}"
    )
    return {
        "rho": float(result["rho"]),
        "eta": float(result["eta"]),
        "delta": float(result["delta"]),
        "delta_levels": result.get("delta_levels", None),
        "top_b": int(result["top_b"]),
        "top_b_levels": result.get("top_b_levels", None),
        "val_micro": float(result["micro"]),
        "val_macro": float(result["macro"]),
        "val_score": float(result["score"]),
        "tuning_info": result.get("tuning_info", ""),
    }


def save_posthoc_baseline_summary(
    summary_dir: str,
    seed: int,
    checkpoint_path: str,
    ctx: "SensitivityContext",
) -> None:
    summary_dir = str(summary_dir or "").strip()
    if not summary_dir:
        return
    out_dir = os.path.join(summary_dir, f"seed{int(seed)}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "posthoc_baseline_summary.json")
    payload = {
        "seed": int(seed),
        "checkpoint_path": checkpoint_path,
        "rho": ctx.base_rho,
        "eta": ctx.base_eta,
        "delta": ctx.base_delta,
        "delta_levels": ctx.base_delta_levels,
        "top_b": ctx.base_top_b,
        "top_b_levels": ctx.base_top_b_levels,
        "val_micro": ctx.baseline_params.get("val_micro"),
        "val_macro": ctx.baseline_params.get("val_macro"),
        "val_score": ctx.baseline_params.get("val_score"),
        "tuning_info": ctx.baseline_params.get("tuning_info", ""),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[Seed {seed}] Saved post-hoc baseline summary: {out_path}")


def ensure_checkpoint(cfg: TrainConfig, resume: bool, force_train: bool) -> str:
    checkpoint_path = checkpoint_path_for(cfg.workdir)
    if resume and not force_train and os.path.exists(checkpoint_path):
        print(f"[Seed {cfg.seed}] Reusing checkpoint: {checkpoint_path}")
        return checkpoint_path

    print(f"[Seed {cfg.seed}] Training once for sensitivity checkpoint...")
    train_main(cfg, scenario_name=f"sensitivity_seed_{cfg.seed}")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Expected checkpoint was not created: {checkpoint_path}")
    return checkpoint_path


def load_hierarchy(cfg: TrainConfig):
    root_names = parse_root_label_names(getattr(cfg, "root_label_name", "Root"))
    if bool(getattr(cfg, "exclude_root_label", False)):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = promote_named_roots(json.load(handle), root_names)
        return parse_label_hierarchy(hjson), root_names
    return load_hierarchy_from_file(cfg.hierarchy_json), root_names


class SensitivityContext:
    def __init__(
        self,
        cfg: TrainConfig,
        hd,
        hierarchy_obj: Hierarchy,
        label_levels: List[int],
        checkpoint: Dict[str, object],
        baseline_params: Dict[str, object],
        prepared_memory_inputs,
        Z_eval: torch.Tensor,
        X_te_dev: torch.Tensor,
        p_cls_te: torch.Tensor,
        y_true: np.ndarray,
        device_str: str,
    ):
        self.cfg = cfg
        self.hd = hd
        self.hierarchy_obj = hierarchy_obj
        self.label_levels = label_levels
        self.checkpoint = checkpoint
        self.baseline_params = baseline_params
        self.prepared_memory_inputs = prepared_memory_inputs
        self.Z_eval = Z_eval
        self.X_te_dev = X_te_dev
        self.p_cls_te = p_cls_te
        self.y_true = y_true
        self.device_str = device_str
        baseline = baseline_params or {}
        self.base_rho = float(baseline.get("rho", checkpoint.get("rho", cfg.rho)))
        self.base_eta = float(baseline.get("eta", checkpoint.get("eta", cfg.eta)))
        self.base_delta = float(baseline.get("delta", checkpoint.get("delta", cfg.delta)))
        self.base_delta_levels = baseline.get("delta_levels", checkpoint.get("delta_levels", None))
        self.base_top_b = int(baseline.get("top_b", checkpoint.get("top_b", cfg.top_b)))
        self.base_top_b_levels = baseline.get("top_b_levels", checkpoint.get("top_b_levels", None))


def build_context(
    cfg: TrainConfig,
    checkpoint_path: str,
    baseline_params: Optional[Dict[str, object]] = None,
) -> SensitivityContext:
    set_seed(cfg.seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    hd, root_names = load_hierarchy(cfg)
    label_levels = [{int(k): int(v) for k, v in hd.levels.items()}.get(i, 1) for i in range(hd.num_labels)]

    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [parse_label_cell(cell) for cell in df_all[cfg.labels_col].tolist()]
    if bool(getattr(cfg, "exclude_root_label", False)):
        all_label_lists = [strip_root_label(labels, root_names) for labels in all_label_lists]
    Y_all = np.array(build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
    train_pool_idx, train_rel_idx, val_rel_idx, _train_idx_np, _val_idx_np, test_idx_np = build_two_stage_split_indices(
        Y_all, cfg
    )
    df_train = df_all.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx_np].reset_index(drop=True)
    Y_tr_full = torch.tensor(Y_all[train_pool_idx], dtype=torch.float32)
    Y_te = torch.tensor(Y_all[test_idx_np], dtype=torch.float32)

    tokenizer_owner = SharedEncoder(build_encoder_config(cfg, device_str))
    label_descs = build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
    train_tokens = tokenize_texts(tokenizer_owner.tokenizer, df_train[cfg.text_col].astype(str).tolist(), cfg.max_len)
    test_tokens = tokenize_texts(tokenizer_owner.tokenizer, df_test[cfg.text_col].astype(str).tolist(), cfg.max_len)
    label_tokens = tokenize_texts(tokenizer_owner.tokenizer, label_descs, cfg.max_len)
    del tokenizer_owner

    enc, clf, checkpoint = load_model_from_checkpoint_for_test(cfg, checkpoint_path, device, device_str)
    enc.eval()
    if clf is not None:
        clf.eval()
    eval_train_tokens = subset_tokens(train_tokens, train_rel_idx)
    val_tokens = subset_tokens(train_tokens, val_rel_idx)
    eval_train_labels = Y_tr_full.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))
    Y_va = Y_tr_full.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))

    hierarchy_obj = Hierarchy(num_labels=hd.num_labels, ancestors=hd.ancestors)
    baseline_params = dict(baseline_params or {})
    if not baseline_params:
        baseline_params = baseline_params_from_holdout(
            cfg=cfg,
            hd=hd,
            hierarchy_obj=hierarchy_obj,
            label_levels=label_levels,
            label_tokens=label_tokens,
            train_tokens=eval_train_tokens,
            Y_tr=eval_train_labels,
            val_tokens=val_tokens,
            Y_va=Y_va,
            enc=enc,
            clf=clf,
            device=device,
            device_str=device_str,
        )

    Z_eval = encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
    X_tr_mem = encode_with_encoder(enc, eval_train_tokens, cfg.batch_size, device)
    prepared_memory_inputs = prepare_memory_inputs(
        X_tr_mem,
        eval_train_labels,
        level_slices=make_level_slices(hd.levels),
    )
    X_te_enc = encode_with_encoder(enc, test_tokens, cfg.batch_size, device)
    X_te_dev = X_te_enc.to(device)
    with torch.no_grad():
        if clf is not None:
            p_cls_te = clf(X_te_dev)["p_cls"]
        else:
            p_cls_te = torch.zeros(X_te_dev.size(0), int(sum(hd.level_sizes)), device=device)

    return SensitivityContext(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        label_levels=label_levels,
        checkpoint=checkpoint,
        baseline_params=baseline_params or {},
        prepared_memory_inputs=prepared_memory_inputs,
        Z_eval=Z_eval,
        X_te_dev=X_te_dev,
        p_cls_te=p_cls_te,
        y_true=(Y_te.cpu().numpy() > 0.5).astype(np.int32),
        device_str=device_str,
    )


def build_memory(ctx: SensitivityContext, rho: float):
    mem_cfg = MemoryConfig(
        backend="faiss_ip",
        top_b=ctx.base_top_b,
        tau_mem=ctx.cfg.tau_mem,
        rho=float(rho),
        device=ctx.device_str,
    )
    return build_memory_store(
        ctx.prepared_memory_inputs,
        ctx.Z_eval,
        mem_cfg,
        rho=float(rho),
    )


def evaluate_scores(
    ctx: SensitivityContext,
    mem,
    top_b: object,
    eta: float,
) -> Tuple[float, float]:
    engine = InferenceEngine(
        InferenceConfig(eta=float(eta), delta=ctx.base_delta, device=ctx.device_str),
        ctx.hierarchy_obj,
    )
    with torch.no_grad():
        s_mem = mem.batch_query(ctx.X_te_dev, top_b=top_b)
    pred = predict_with_strategy(
        s_mem=s_mem,
        p_cls=ctx.p_cls_te,
        engine=engine,
        cfg=ctx.cfg,
        eta_override=float(eta),
        delta_override=ctx.base_delta,
        label_levels=ctx.label_levels,
        delta_levels_override=ctx.base_delta_levels,
    )
    y_pred = pred.cpu().numpy().astype(np.int32)
    return micro_f1(ctx.y_true, y_pred), macro_f1(ctx.y_true, y_pred)


def result_row(seed: int, param_name: str, param_value: object, micro: float, macro: float) -> Dict[str, str]:
    return {
        "seed": str(int(seed)),
        "param_name": str(param_name),
        "param_value": str(param_value),
        "micro": f"{float(micro):.4f}",
        "macro": f"{float(macro):.4f}",
    }


def run_seed(
    cfg: TrainConfig,
    checkpoint_path: str,
    baseline_params: Dict[str, object],
    baseline_summary_dir: str,
    rho_values: Iterable[float],
    top_b_values: Iterable[int],
    eta_values: Iterable[float],
    existing_rows: Dict[Tuple[int, str, str], Dict[str, str]],
    resume: bool,
    overwrite_existing: bool,
) -> None:
    seed = int(cfg.seed)
    needed = [
        *[("rho", format_float(v)) for v in rho_values],
        *[("top_b", str(int(v))) for v in top_b_values],
        *[("eta", format_float(v)) for v in eta_values],
    ]
    if resume and not overwrite_existing and all(row_key(seed, name, value) in existing_rows for name, value in needed):
        print(f"[Seed {seed}] Sensitivity rows already complete; skipping evaluation.")
        return

    ctx = build_context(cfg, checkpoint_path, baseline_params=baseline_params)
    save_posthoc_baseline_summary(baseline_summary_dir, seed, checkpoint_path, ctx)
    base_mem = None

    for rho in rho_values:
        value = format_float(rho)
        key = row_key(seed, "rho", value)
        if resume and not overwrite_existing and key in existing_rows:
            continue
        mem = build_memory(ctx, float(rho))
        micro, macro = evaluate_scores(ctx, mem, ctx.base_top_b_levels or ctx.base_top_b, ctx.base_eta)
        existing_rows[key] = result_row(seed, "rho", value, micro, macro)
        print(f"[Seed {seed}] rho={value} micro={micro:.4f} macro={macro:.4f}")

    if base_mem is None:
        base_mem = build_memory(ctx, ctx.base_rho)

    for top_b in top_b_values:
        value = str(int(top_b))
        key = row_key(seed, "top_b", value)
        if resume and not overwrite_existing and key in existing_rows:
            continue
        micro, macro = evaluate_scores(ctx, base_mem, int(top_b), ctx.base_eta)
        existing_rows[key] = result_row(seed, "top_b", value, micro, macro)
        print(f"[Seed {seed}] top_b={value} micro={micro:.4f} macro={macro:.4f}")

    base_scores = None
    for eta in eta_values:
        value = format_float(eta)
        key = row_key(seed, "eta", value)
        if resume and not overwrite_existing and key in existing_rows:
            continue
        if base_scores is None:
            with torch.no_grad():
                top_b_query = ctx.base_top_b_levels or ctx.base_top_b
                s_mem_base = base_mem.batch_query(ctx.X_te_dev, top_b=top_b_query)
            base_scores = s_mem_base
        engine = InferenceEngine(
            InferenceConfig(eta=float(eta), delta=ctx.base_delta, device=ctx.device_str),
            ctx.hierarchy_obj,
        )
        pred = predict_with_strategy(
            s_mem=base_scores,
            p_cls=ctx.p_cls_te,
            engine=engine,
            cfg=ctx.cfg,
            eta_override=float(eta),
            delta_override=ctx.base_delta,
            label_levels=ctx.label_levels,
            delta_levels_override=ctx.base_delta_levels,
        )
        y_pred = pred.cpu().numpy().astype(np.int32)
        micro = micro_f1(ctx.y_true, y_pred)
        macro = macro_f1(ctx.y_true, y_pred)
        existing_rows[key] = result_row(seed, "eta", value, micro, macro)
        print(f"[Seed {seed}] eta={value} micro={micro:.4f} macro={macro:.4f}")


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(args)
    rows = load_existing_rows(args.output_csv) if args.resume else {}

    for seed in seeds:
        seed_workdir = os.path.join(args.workdir, f"seed_{seed}")
        cfg = replace(TrainConfig(), seed=int(seed), workdir=seed_workdir)
        checkpoint_path = resolve_checkpoint_path(
            cfg,
            checkpoint_root=args.checkpoint_root,
            checkpoint_scenario=args.checkpoint_scenario,
            resume=args.resume,
            force_train=args.force_train,
        )
        baseline_params = load_posthoc_baseline_params(
            args.posthoc_summary_root,
            int(seed),
            args.posthoc_summary_scenario,
        )
        run_seed(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            baseline_params=baseline_params,
            baseline_summary_dir=args.baseline_summary_dir,
            rho_values=args.rho_values,
            top_b_values=args.top_b_values,
            eta_values=args.eta_values,
            existing_rows=rows,
            resume=args.resume,
            overwrite_existing=args.overwrite_existing,
        )
        write_rows(args.output_csv, rows)

    print(f"[Done] Sensitivity results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
