#!/usr/bin/env python3
import argparse
import csv
import os
import re
import time
from dataclasses import replace
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCENARIO_SPECS: List[Tuple[str, str, Dict[str, object]]] = [
    (
        "global_only",
        "HMCN-Global only",
        {"use_memory": False, "use_local_branch": False, "use_global_branch": True},
    ),
    (
        "local_only",
        "HMCN-Local only",
        {"use_memory": False, "use_local_branch": True, "use_global_branch": False},
    ),
    (
        "hmcn_direct_sum",
        "HMCN (direct-sum fusion)",
        {
            "use_memory": False,
            "use_local_branch": True,
            "use_global_branch": True,
            "fusion_mode": "direct_sum",
        },
    ),
    (
        "hmcn_residual",
        "HMCN (w/ residual fusion)",
        {
            "use_memory": False,
            "use_local_branch": True,
            "use_global_branch": True,
            "fusion_mode": "residual",
        },
    ),
    (
        "raehmc_sum_posthoc",
        "RAE-HMC (direct-sum backbone + post-hoc retrieval)",
        {
            "use_memory": True,
            "retrieval_protocol": "post_hoc",
            "use_local_branch": True,
            "use_global_branch": True,
            "fusion_mode": "direct_sum",
        },
    ),
    (
        "raehmc_residual_posthoc",
        "RAE-HMC (residual backbone + post-hoc retrieval)",
        {
            "use_memory": True,
            "retrieval_protocol": "post_hoc",
            "use_local_branch": True,
            "use_global_branch": True,
            "fusion_mode": "residual",
        },
    ),
]

SCENARIO_COLUMNS = {
    "global_only": ("global_mic", "global_mac"),
    "local_only": ("local_mic", "local_mac"),
    "hmcn_direct_sum": ("sum_mic", "sum_mac"),
    "hmcn_residual": ("residual_mic", "residual_mac"),
    "raehmc_sum_posthoc": ("raehmc_Sum_mic", "raehmc_Sum_mac"),
    "raehmc_residual_posthoc": ("raehmc_Residual_mic", "raehmc_Residual_mac"),
}

CSV_FIELDNAMES: List[str] = ["seed"]
for _scenario_key, _display_name, _overrides in SCENARIO_SPECS:
    CSV_FIELDNAMES.extend(SCENARIO_COLUMNS[_scenario_key])

DEFAULT_WORKDIR = "./outputs/ablation_module"
LEGACY_DISPLAY_TO_KEY = {
    "Global Semantic Branch": "global_only",
    "Local Hierarchical Branch": "local_only",
    "Global + Local Branches": "hmcn_residual",
    "Full Model": "raehmc_residual_posthoc",
    "HMCN-Global only": "global_only",
    "HMCN-Local only": "local_only",
    "HMCN (direct-sum fusion)": "hmcn_direct_sum",
    "HMCN (w/ Logit residual)": "hmcn_residual",
    "HMCN (w/ Prob residual)": "hmcn_residual",
    "HMCN + Retrieval (w/ Logit residual)": "raehmc_residual_posthoc",
    "HMCN + Retrieval (w/ Prob residual)": "raehmc_residual_posthoc",
    "HMCN (w/ residual fusion)": "hmcn_residual",
    "RAE-HMC": "raehmc_residual_posthoc",
    "RAE-HMC (post-hoc retrieval, w/ residual)": "raehmc_residual_posthoc",
    "RAE-HMC (per-epoch retrieval, w/ residual)": "raehmc_residual_posthoc",
}

POSTHOC_BACKBONES = {
    "raehmc_sum_posthoc": "hmcn_direct_sum",
    "raehmc_residual_posthoc": "hmcn_residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAE-HMC ablation experiments across multiple seeds."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seed list. If omitted, uses the inclusive seed range.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=41,
        help="Inclusive start seed when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=70,
        help="Inclusive end seed when --seeds is not provided.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=DEFAULT_WORKDIR,
        help="Base output directory. Seed-wise results are written to <workdir>/ablation_paired.csv.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse rows from an existing ablation_paired.csv and skip seed/scenario cells "
            "whose micro/macro metrics are already present."
        ),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=(
            "Optional scenario keys to run, e.g. hmcn_residual raehmc_residual_posthoc. "
            "Valid keys: " + ", ".join(key for key, _, _ in SCENARIO_SPECS)
        ),
    )
    parser.add_argument(
        "--overwrite-selected",
        action="store_true",
        help=(
            "When used with --resume and --scenarios, clear the selected scenario cells "
            "for the requested seeds before running."
        ),
    )
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return sorted(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def resolve_scenarios(args: argparse.Namespace) -> List[Tuple[str, str, Dict[str, object]]]:
    if not args.scenarios:
        return SCENARIO_SPECS
    available = {key: (key, display_name, overrides) for key, display_name, overrides in SCENARIO_SPECS}
    selected: List[Tuple[str, str, Dict[str, object]]] = []
    unknown: List[str] = []
    for raw_key in args.scenarios:
        key = str(raw_key).strip()
        if key not in available:
            unknown.append(key)
            continue
        selected.append(available[key])
    if unknown:
        raise ValueError(
            "Unknown scenario key(s): "
            + ", ".join(unknown)
            + ". Valid keys: "
            + ", ".join(available)
        )
    return selected

SEED_HEADER_RE = re.compile(r"^\[Seed (?P<seed>\d+)\]$")
MODE_LINE_RE = re.compile(r"^(?:\[Resume\]\s+)?mode=(?P<mode>\w+),\s+seeds=")
LEGACY_RESULT_LINE_RE = re.compile(
    r"^  - (?P<scenario>.*?): eta=(?P<eta>.*?), delta=(?P<delta>.*?), "
    r"rho=(?P<rho>.*?), top_b=(?P<top_b>.*?), micro-F1=(?P<micro>[0-9.]+), "
    r"macro-F1\(all\)=(?P<macro>[0-9.]+)(?:, time=(?P<time>[0-9:]+))?$"
)


def scenario_column(scenario_key: str, suffix: str) -> str:
    if suffix == "micro":
        return SCENARIO_COLUMNS[scenario_key][0]
    if suffix == "macro":
        return SCENARIO_COLUMNS[scenario_key][1]
    raise ValueError(f"Unsupported metric suffix: {suffix}")


def format_delta_display(delta: object, delta_levels: Optional[Dict[int, float]] = None) -> str:
    if delta_levels:
        levels_sorted = sorted((int(k), float(v)) for k, v in delta_levels.items())
        return "{" + ", ".join([f"L{lvl}={val:.2f}" for lvl, val in levels_sorted]) + "}"
    return f"{float(delta):.2f}"


def format_top_b_display(top_b: object, top_b_levels: Optional[Sequence[int]] = None) -> str:
    if top_b_levels:
        levels_sorted = [f"L{idx + 1}={int(val)}" for idx, val in enumerate(top_b_levels)]
        return "{" + ", ".join(levels_sorted) + "}"
    return f"{int(top_b)}"


def format_elapsed_time(seconds: object) -> str:
    total_seconds = int(round(float(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def empty_result_row(seed: int) -> Dict[str, str]:
    row = {fieldname: "" for fieldname in CSV_FIELDNAMES}
    row["seed"] = str(seed)
    return row


def initialize_blank_csv(csv_path: str, seeds: Sequence[int]) -> None:
    rows_by_seed = {int(seed): empty_result_row(int(seed)) for seed in seeds}
    write_result_csv(csv_path, rows_by_seed)


def load_existing_rows(csv_path: str) -> Dict[int, Dict[str, str]]:
    rows_by_seed: Dict[int, Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        return rows_by_seed

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            seed_text = (raw_row.get("seed") or "").strip()
            if not seed_text:
                continue
            seed = int(seed_text)
            row = empty_result_row(seed)
            for fieldname in CSV_FIELDNAMES:
                if fieldname == "seed":
                    continue
                if fieldname in raw_row and raw_row[fieldname] is not None:
                    row[fieldname] = raw_row[fieldname]
            rows_by_seed[seed] = row
    return rows_by_seed


def write_result_csv(csv_path: str, rows_by_seed: Dict[int, Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for seed in sorted(rows_by_seed):
            row = empty_result_row(seed)
            row.update(rows_by_seed[seed])
            row["seed"] = str(seed)
            for fieldname in CSV_FIELDNAMES:
                if fieldname.endswith(("_mic", "_mac")) and row.get(fieldname, "").strip():
                    row[fieldname] = f"{float(row[fieldname]):.4f}"
            writer.writerow(row)


def scenario_complete(row: Dict[str, str], scenario_key: str) -> bool:
    return bool(row.get(scenario_column(scenario_key, "micro"), "").strip()) and bool(
        row.get(scenario_column(scenario_key, "macro"), "").strip()
    )


def metric_values(
    rows_by_seed: Dict[int, Dict[str, str]],
    scenario_key: str,
    metric: str,
) -> List[float]:
    values: List[float] = []
    column = scenario_column(scenario_key, metric)
    for seed in sorted(rows_by_seed):
        value = rows_by_seed[seed].get(column, "").strip()
        if not value:
            continue
        values.append(float(value))
    return values


def update_row_from_result(row: Dict[str, str], item: Dict[str, object]) -> None:
    scenario_key = str(item["scenario_key"])
    row[scenario_column(scenario_key, "micro")] = f"{float(item['micro']):.4f}"
    row[scenario_column(scenario_key, "macro")] = f"{float(item['macro_all']):.4f}"


def load_legacy_txt_rows(
    path: str,
    scenario_specs: List[Tuple[str, str, Dict[str, object]]],
) -> Dict[int, Dict[str, str]]:
    rows_by_seed: Dict[int, Dict[str, str]] = {}
    current_mode = ""
    current_seed = None

    if not os.path.exists(path):
        return rows_by_seed

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            mode_match = MODE_LINE_RE.match(line)
            if mode_match:
                current_mode = mode_match.group("mode")
                continue

            seed_match = SEED_HEADER_RE.match(line)
            if seed_match:
                current_seed = int(seed_match.group("seed"))
                row = rows_by_seed.setdefault(current_seed, empty_result_row(current_seed))
                continue

            if line.startswith("[Aggregate]"):
                current_seed = None
                continue
            if current_seed is None:
                continue

            result_match = LEGACY_RESULT_LINE_RE.match(line)
            if not result_match:
                continue
            scenario_key = LEGACY_DISPLAY_TO_KEY.get(result_match.group("scenario"))
            if scenario_key is None:
                continue

            row = rows_by_seed.setdefault(current_seed, empty_result_row(current_seed))

            row[scenario_column(scenario_key, "micro")] = f"{float(result_match.group('micro')):.4f}"
            row[scenario_column(scenario_key, "macro")] = f"{float(result_match.group('macro')):.4f}"

    return rows_by_seed


def convert_legacy_summary_to_csv(legacy_txt_path: str, csv_path: str) -> int:
    rows_by_seed = load_legacy_txt_rows(legacy_txt_path, SCENARIO_SPECS)
    if not rows_by_seed:
        raise ValueError(f"No seed-wise results found in {legacy_txt_path}")
    write_result_csv(csv_path, rows_by_seed)
    return len(rows_by_seed)


def format_summary_line(item: Dict[str, object]) -> str:
    use_memory = bool(item.get("use_memory", True))
    use_global = bool(item.get("use_global_branch", True))
    use_local = bool(item.get("use_local_branch", True))
    classifier_on = use_global or use_local
    fusion_on = use_memory and classifier_on

    eta_print = f"{float(item.get('eta', 0.0)):.2f}" if fusion_on else "N/A"
    delta_print = (
        format_delta_display(item.get("delta", 0.0), item.get("delta_levels"))
        if (use_memory or classifier_on)
        else "N/A"
    )
    rho_val = item.get("rho", None)
    rho_print = f"{float(rho_val):.2f}" if (use_memory and rho_val is not None) else "N/A"
    top_b_val = item.get("top_b", None)
    top_b_levels = item.get("top_b_levels", None)
    top_b_print = (
        format_top_b_display(top_b_val if top_b_val is not None else 0, top_b_levels)
        if use_memory and (top_b_val is not None or top_b_levels is not None)
        else "N/A"
    )
    runtime_seconds = item.get("runtime_seconds", None)
    runtime_text = (
        f", time={format_elapsed_time(runtime_seconds)}"
        if runtime_seconds is not None
        else ""
    )
    return (
        f"  - {item['scenario']}: eta={eta_print}, delta={delta_print}, rho={rho_print}, "
        f"top_b={top_b_print}, micro-F1={float(item['micro']):.4f}, "
        f"macro-F1(all)={float(item['macro_all']):.4f}{runtime_text}"
    )


def summarize_scenario_setup(cfg: object) -> str:
    enabled_parts: List[str] = []
    if bool(getattr(cfg, "use_global_branch", False)):
        enabled_parts.append("global")
    if bool(getattr(cfg, "use_local_branch", False)):
        enabled_parts.append("local")
    if bool(getattr(cfg, "use_memory", False)):
        enabled_parts.append("retrieval")
    if not enabled_parts:
        enabled_parts.append("none")

    summary = "/".join(enabled_parts)
    if bool(getattr(cfg, "use_global_branch", False)) and bool(getattr(cfg, "use_local_branch", False)):
        summary += f", fusion={str(getattr(cfg, 'fusion_mode', 'residual')).strip().lower()}"
    return summary


def print_scenario_banner(
    seed: int,
    display_name: str,
    scenario_key: str,
    cfg: object,
    action: str,
) -> None:
    print()
    print("=" * 88)
    print(f"[Seed {seed}] {action}: {display_name}")
    print(f"scenario_key={scenario_key}, setup={summarize_scenario_setup(cfg)}")
    print(f"workdir={getattr(cfg, 'workdir', '')}")
    print("=" * 88, flush=True)


def checkpoint_path(base_workdir: str, seed: int, scenario_key: str) -> str:
    return os.path.join(
        base_workdir,
        "ablation_runs",
        f"seed{seed}",
        scenario_key,
        "best_model_holdout.pt",
    )


def resolve_backbone_checkpoint(
    base_workdir: str,
    seed: int,
    backbone_scenario_key: str,
) -> Tuple[str, str]:
    checkpoint = checkpoint_path(base_workdir, seed, backbone_scenario_key)
    resolved_key = backbone_scenario_key
    if os.path.exists(checkpoint):
        return checkpoint, resolved_key

    legacy_backbone_aliases = {}
    for alias in legacy_backbone_aliases.get(backbone_scenario_key, []):
        alias_checkpoint = checkpoint_path(base_workdir, seed, alias)
        if os.path.exists(alias_checkpoint):
            return alias_checkpoint, alias

    return checkpoint, resolved_key


def run_posthoc_retrieval_eval(
    *,
    base_cfg,
    base_workdir: str,
    seed: int,
    cfg,
    scenario_name: str,
    backbone_scenario_key: str = "hmcn_residual",
) -> Dict[str, object]:
    import json

    import numpy as np
    import pandas as pd
    import torch

    from train_rae_hmc import (
        Hierarchy,
        MemoryConfig,
        SharedEncoder,
        build_encoder_config,
        build_label_descriptions,
        build_multi_hot_Y,
        build_two_stage_split_indices,
        evaluate_model_on_test_split,
        load_hierarchy_from_file,
        load_model_from_checkpoint_for_test,
        make_level_slices,
        parse_label_cell,
        parse_label_hierarchy,
        parse_root_label_names,
        promote_named_roots,
        set_seed,
        strip_root_label,
        subset_tokens,
        tokenize_texts,
        tune_validation_strategy,
    )

    start_time = time.perf_counter()
    os.makedirs(cfg.workdir, exist_ok=True)
    set_seed(seed)

    checkpoint, resolved_backbone_key = resolve_backbone_checkpoint(
        base_workdir, seed, backbone_scenario_key
    )
    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"Post-hoc retrieval requires an existing backbone checkpoint: {checkpoint}"
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    data_cfg = replace(base_cfg, seed=seed)

    root_names = parse_root_label_names(getattr(data_cfg, "root_label_name", "Root"))
    if bool(getattr(data_cfg, "exclude_root_label", False)):
        with open(data_cfg.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = json.load(handle)
        hjson = promote_named_roots(hjson, root_names)
        hd = parse_label_hierarchy(hjson)
    else:
        hd = load_hierarchy_from_file(data_cfg.hierarchy_json)

    level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
    label_levels = [level_lookup.get(i, 1) for i in range(hd.num_labels)]
    df_all = pd.read_csv(data_cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [parse_label_cell(s) for s in df_all[data_cfg.labels_col].tolist()]
    if bool(getattr(data_cfg, "exclude_root_label", False)):
        all_label_lists = [strip_root_label(labels, root_names) for labels in all_label_lists]

    Y_all = np.array(
        build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True)
    )
    train_pool_idx, train_rel_idx, val_rel_idx, _, _, test_idx_np = build_two_stage_split_indices(
        Y_all, data_cfg
    )
    df_train = df_all.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx_np].reset_index(drop=True)
    Y_tr_full = torch.tensor(Y_all[train_pool_idx], dtype=torch.float32)
    Y_te = torch.tensor(Y_all[test_idx_np], dtype=torch.float32)

    tokenizer_encoder = SharedEncoder(build_encoder_config(data_cfg, device_str))
    label_descs = build_label_descriptions(hd, getattr(data_cfg, "label_path_depth", 1))
    train_tokens_full = tokenize_texts(
        tokenizer_encoder.tokenizer,
        df_train[data_cfg.text_col].astype(str).tolist(),
        data_cfg.max_len,
    )
    test_tokens = tokenize_texts(
        tokenizer_encoder.tokenizer,
        df_test[data_cfg.text_col].astype(str).tolist(),
        data_cfg.max_len,
    )
    label_tokens = tokenize_texts(tokenizer_encoder.tokenizer, label_descs, data_cfg.max_len)
    del tokenizer_encoder

    hierarchy_obj = Hierarchy(num_labels=hd.num_labels, ancestors=hd.ancestors)
    level_slices = make_level_slices(hd.levels)
    mem_cfg = MemoryConfig(
        backend="faiss_ip",
        top_b=data_cfg.top_b,
        tau_mem=data_cfg.tau_mem,
        rho=data_cfg.rho,
        device=device_str,
    )

    train_tokens = subset_tokens(train_tokens_full, train_rel_idx)
    val_tokens = subset_tokens(train_tokens_full, val_rel_idx)
    Y_tr = Y_tr_full.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))
    Y_va = Y_tr_full.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))

    enc, clf, _ = load_model_from_checkpoint_for_test(
        cfg=cfg,
        checkpoint_path=checkpoint,
        device=device,
        device_str=device_str,
    )

    dynamic_result = tune_validation_strategy(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
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

    mem_cfg_eval = replace(
        mem_cfg,
        rho=float(dynamic_result["rho"]),
        top_b=int(dynamic_result["top_b"]),
    )
    test_result = evaluate_model_on_test_split(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        label_levels=label_levels,
        label_tokens=label_tokens,
        train_tokens_for_memory=train_tokens,
        Y_train_for_memory=Y_tr,
        test_tokens=test_tokens,
        Y_te=Y_te,
        enc=enc,
        clf=clf,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg_eval,
        eta_final=float(dynamic_result["eta"]),
        delta_final=float(dynamic_result["delta"]),
        top_b_final=int(dynamic_result["top_b"]),
        top_b_levels_final=dynamic_result.get("top_b_levels"),
        delta_levels_final=dynamic_result.get("delta_levels"),
    )

    runtime_seconds = time.perf_counter() - start_time
    summary_payload = {
        "backbone_scenario": resolved_backbone_key,
        "checkpoint_path": checkpoint,
        "eta": float(dynamic_result["eta"]),
        "delta": float(dynamic_result["delta"]),
        "delta_levels": dynamic_result.get("delta_levels"),
        "rho": float(dynamic_result["rho"]),
        "top_b": int(dynamic_result["top_b"]),
        "top_b_levels": dynamic_result.get("top_b_levels"),
        "val_micro": float(dynamic_result["micro"]),
        "val_macro": float(dynamic_result["macro"]),
        "test_micro": float(test_result["micro"]),
        "test_macro": float(test_result["macro_all"]),
        "runtime_seconds": runtime_seconds,
    }
    with open(
        os.path.join(cfg.workdir, "posthoc_eval_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    top_b_text = format_top_b_display(
        summary_payload["top_b"],
        summary_payload["top_b_levels"],
    )
    print(
        f"[Post-hoc] Backbone={resolved_backbone_key}, checkpoint={checkpoint}, "
        f"eta={summary_payload['eta']:.2f}, delta={summary_payload['delta']:.2f}, "
        f"rho={summary_payload['rho']:.2f}, top_b={top_b_text}"
    )
    print(
        f"[Post-hoc TEST] micro-F1={summary_payload['test_micro']:.4f}, "
        f"macro-F1(all)={summary_payload['test_macro']:.4f}"
    )

    return {
        "scenario": scenario_name,
        "eta": summary_payload["eta"],
        "delta": summary_payload["delta"],
        "delta_levels": summary_payload["delta_levels"],
        "rho": summary_payload["rho"],
        "top_b": summary_payload["top_b"],
        "top_b_levels": summary_payload["top_b_levels"],
        "micro": summary_payload["test_micro"],
        "macro_all": summary_payload["test_macro"],
        "runtime_seconds": runtime_seconds,
        "runtime_text": format_elapsed_time(runtime_seconds),
        "use_memory": True,
        "use_global_branch": True,
        "use_local_branch": True,
    }


def iter_seed_runs(
    base_cfg,
    base_workdir: str,
    seed: int,
    scenario_specs: List[Tuple[str, str, Dict[str, object]]],
) -> Iterable[Tuple[str, str, object]]:
    for scenario_key, display_name, overrides in scenario_specs:
        yield scenario_key, display_name, replace(
            base_cfg,
            **overrides,
            seed=seed,
            workdir=os.path.join(base_workdir, "ablation_runs", f"seed{seed}", scenario_key),
        )


def main() -> None:
    args = parse_args()
    from train_rae_hmc import TrainConfig, main as train_main

    seeds = resolve_seeds(args)
    scenario_specs = resolve_scenarios(args)
    base_cfg = TrainConfig()
    base_workdir = args.workdir
    os.makedirs(base_workdir, exist_ok=True)
    csv_path = os.path.join(base_workdir, "ablation_paired.csv")
    if args.scenarios and not args.resume and os.path.exists(csv_path):
        raise ValueError(
            "Using --scenarios with an existing ablation_paired.csv requires --resume "
            "to avoid wiping unrelated columns. Add --overwrite-selected if you want to rerun selected cells."
        )
    rows_by_seed = load_existing_rows(csv_path) if args.resume else {}
    for seed in seeds:
        rows_by_seed.setdefault(seed, empty_result_row(seed))
    if args.overwrite_selected:
        if not args.scenarios:
            raise ValueError("--overwrite-selected requires --scenarios.")
        if not args.resume:
            raise ValueError("--overwrite-selected requires --resume.")
        for seed in seeds:
            row = rows_by_seed.setdefault(seed, empty_result_row(seed))
            for scenario_key, _, _ in scenario_specs:
                row[scenario_column(scenario_key, "micro")] = ""
                row[scenario_column(scenario_key, "macro")] = ""
    write_result_csv(csv_path, rows_by_seed)

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"\n[{seed_idx}/{len(seeds)}] seed={seed}")

        seed_summary: List[Dict[str, object]] = []
        skipped_scenarios: Set[str] = set()
        row = rows_by_seed[seed]
        for scenario_key, display_name, cfg in iter_seed_runs(
            base_cfg, base_workdir, seed, scenario_specs
        ):
            if scenario_complete(row, scenario_key):
                skipped_scenarios.add(display_name)
                print_scenario_banner(seed, display_name, scenario_key, cfg, "SKIP")
                print(f"already in {csv_path}", flush=True)
                continue
            print_scenario_banner(seed, display_name, scenario_key, cfg, "RUN")
            if scenario_key in POSTHOC_BACKBONES:
                res = run_posthoc_retrieval_eval(
                    base_cfg=base_cfg,
                    base_workdir=base_workdir,
                    seed=seed,
                    cfg=cfg,
                    scenario_name=scenario_key,
                    backbone_scenario_key=POSTHOC_BACKBONES[scenario_key],
                )
            else:
                res = train_main(cfg, scenario_name=scenario_key)
            if res is None:
                raise RuntimeError(
                    f"Scenario {scenario_key} for seed {seed} returned None unexpectedly."
                )
            res["scenario_key"] = scenario_key
            res["scenario"] = display_name
            seed_summary.append(res)
            update_row_from_result(row, res)
            write_result_csv(csv_path, rows_by_seed)

        if seed_summary:
            print("[Ablation Summary]")
        for item in seed_summary:
            print(format_summary_line(item))
        if skipped_scenarios and not seed_summary:
            print(
                "[Ablation Summary] all requested scenarios for this seed were already complete."
            )

    print("\n[Aggregate]")
    for scenario_key, display_name, _ in scenario_specs:
        micro_vals = metric_values(rows_by_seed, scenario_key, "micro")
        macro_vals = metric_values(rows_by_seed, scenario_key, "macro")
        if not micro_vals or not macro_vals:
            continue
        micro_mean = fmean(micro_vals)
        macro_mean = fmean(macro_vals)
        micro_std = pstdev(micro_vals) if len(micro_vals) > 1 else 0.0
        macro_std = pstdev(macro_vals) if len(macro_vals) > 1 else 0.0
        print(
            f"  - {display_name}: micro-F1={micro_mean:.4f} ± {micro_std:.4f}, "
            f"macro-F1(all)={macro_mean:.4f} ± {macro_std:.4f}, n={len(micro_vals)}"
        )

    print(f"\n[Done] Ablation paired CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
