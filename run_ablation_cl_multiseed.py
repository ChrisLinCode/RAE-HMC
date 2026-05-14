#!/usr/bin/env python3
import argparse
import csv
import os
from dataclasses import replace
from statistics import fmean, pstdev
from typing import Dict, Iterable, List, Optional, Set, Tuple


SCENARIO_SPECS: List[Tuple[str, str, Dict[str, object]]] = [
    (
        "raehmc_no_cl",
        "RAE-HMC (w/o CL)",
        {"use_sample_cl": False, "use_hnm_cl": False},
    ),
    (
        "raehmc_ss_cl",
        "RAE-HMC (w/ L_SS)",
        {"use_sample_cl": True, "use_hnm_cl": False},
    ),
    (
        "raehmc_hnm_cl",
        "RAE-HMC (w/ L_HNM)",
        {"use_sample_cl": False, "use_hnm_cl": True},
    ),
    (
        "raehmc_cl",
        "RAE-HMC (w/ L_SS+L_HNM)",
        {"use_sample_cl": True, "use_hnm_cl": True},
    ),
]

CSV_FIELDNAMES: List[str] = ["seed"]
for _scenario_key, _display_name, _overrides in SCENARIO_SPECS:
    CSV_FIELDNAMES.extend([f"{_scenario_key}_micro", f"{_scenario_key}_macro"])

DEFAULT_WORKDIR = "./outputs/ablation_cl"
DEFAULT_CSV_NAME = "ablation_cl_paired.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAE-HMC contrastive-loss ablation experiments across multiple seeds."
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
        help=f"Base output directory. Seed-wise results are written to <workdir>/{DEFAULT_CSV_NAME}.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            f"Reuse rows from an existing {DEFAULT_CSV_NAME} and skip seed/scenario cells "
            "whose micro/macro metrics are already present."
        ),
    )
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return sorted(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def scenario_column(scenario_key: str, suffix: str) -> str:
    return f"{scenario_key}_{suffix}"


def empty_result_row(seed: int) -> Dict[str, str]:
    row = {fieldname: "" for fieldname in CSV_FIELDNAMES}
    row["seed"] = str(seed)
    return row


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
                if fieldname.endswith(("_micro", "_macro")) and row.get(fieldname, "").strip():
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


def format_summary_line(item: Dict[str, object]) -> str:
    return (
        f"  - {item['scenario']}: "
        f"sample_cl={'on' if bool(item.get('use_sample_cl', False)) else 'off'}, "
        f"hnm_cl={'on' if bool(item.get('use_hnm_cl', False)) else 'off'}, "
        f"micro-F1={float(item['micro']):.4f}, "
        f"macro-F1(all)={float(item['macro_all']):.4f}"
    )


def summarize_scenario_setup(cfg: object) -> str:
    return (
        "global/local/retrieval, "
        f"sample_cl={'on' if bool(getattr(cfg, 'use_sample_cl', False)) else 'off'}, "
        f"hnm_cl={'on' if bool(getattr(cfg, 'use_hnm_cl', False)) else 'off'}, "
        f"residual={'on' if bool(getattr(cfg, 'use_fusion_residual', True)) else 'off'}"
    )


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


def iter_seed_runs(
    base_cfg,
    base_workdir: str,
    seed: int,
) -> Iterable[Tuple[str, str, object]]:
    for scenario_key, display_name, overrides in SCENARIO_SPECS:
        yield scenario_key, display_name, replace(
            base_cfg,
            **overrides,
            seed=seed,
            workdir=os.path.join(base_workdir, "ablation_cl_runs", f"seed{seed}", scenario_key),
        )


def main() -> None:
    args = parse_args()
    from train_rae_hmc import TrainConfig, main as train_main

    seeds = resolve_seeds(args)
    base_cfg = TrainConfig()
    base_workdir = args.workdir
    os.makedirs(base_workdir, exist_ok=True)
    csv_path = os.path.join(base_workdir, DEFAULT_CSV_NAME)
    rows_by_seed = load_existing_rows(csv_path) if args.resume else {}
    for seed in seeds:
        rows_by_seed.setdefault(seed, empty_result_row(seed))
    write_result_csv(csv_path, rows_by_seed)

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"\n[{seed_idx}/{len(seeds)}] seed={seed}")

        seed_summary: List[Dict[str, object]] = []
        skipped_scenarios: Set[str] = set()
        row = rows_by_seed[seed]
        for scenario_key, display_name, cfg in iter_seed_runs(
            base_cfg, base_workdir, seed
        ):
            if scenario_complete(row, scenario_key):
                skipped_scenarios.add(display_name)
                print_scenario_banner(seed, display_name, scenario_key, cfg, "SKIP")
                print(f"already in {csv_path}", flush=True)
                continue
            print_scenario_banner(seed, display_name, scenario_key, cfg, "RUN")
            res = train_main(cfg, scenario_name=scenario_key)
            if res is None:
                raise RuntimeError(
                    f"Scenario {scenario_key} for seed {seed} returned None unexpectedly."
                )
            res["scenario_key"] = scenario_key
            res["scenario"] = display_name
            res["use_sample_cl"] = bool(getattr(cfg, "use_sample_cl", False))
            res["use_hnm_cl"] = bool(getattr(cfg, "use_hnm_cl", False))
            seed_summary.append(res)
            update_row_from_result(row, res)
            write_result_csv(csv_path, rows_by_seed)

        if seed_summary:
            print("[CL Ablation Summary]")
        for item in seed_summary:
            print(format_summary_line(item))
        if skipped_scenarios and not seed_summary:
            print(
                "[CL Ablation Summary] all requested scenarios for this seed were already complete."
            )

    print("\n[Aggregate]")
    for scenario_key, display_name, _ in SCENARIO_SPECS:
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

    print(f"\n[Done] CL ablation paired CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
