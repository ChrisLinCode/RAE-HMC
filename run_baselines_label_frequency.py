#!/usr/bin/env python3
"""Evaluate baseline checkpoints by Head/Mid/Tail label frequency segments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

import run_baselines_label_depth as depth


SEGMENT_ORDER = ["head", "mid", "tail"]
SEGMENT_DISPLAY = {"head": "Head", "mid": "Mid", "tail": "Tail"}


FREQUENCY_HELPER_INLINE = r"""
SEGMENT_ORDER = ["head", "mid", "tail"]
SEGMENT_DISPLAY = {"head": "Head", "mid": "Mid", "tail": "Tail"}


def frequency_rule(segment, head_min_freq, mid_min_freq):
    head_min_freq = int(head_min_freq)
    mid_min_freq = int(mid_min_freq)
    if segment == "head":
        return f">{head_min_freq}"
    if segment == "mid":
        return f"{mid_min_freq}-{head_min_freq}"
    if segment == "tail":
        return f"<{mid_min_freq}"
    return ""


def frequency_segment(count, head_min_freq, mid_min_freq):
    count = int(count)
    if count > int(head_min_freq):
        return "head"
    if count >= int(mid_min_freq):
        return "mid"
    return "tail"


def frequency_rows(y_true, y_pred, train_freq, head_min_freq, mid_min_freq):
    if int(mid_min_freq) > int(head_min_freq):
        raise ValueError("mid_min_freq must be <= head_min_freq.")
    train_freq = np.asarray(train_freq, dtype=np.int64)
    rows = []
    for segment in SEGMENT_ORDER:
        label_ids = [
            int(label_id)
            for label_id, count in enumerate(train_freq.tolist())
            if frequency_segment(count, head_min_freq, mid_min_freq) == segment
        ]
        cols = np.asarray(label_ids, dtype=int)
        if len(cols) == 0:
            yt = y_true[:, :0].astype(np.int32)
            yp = y_pred[:, :0].astype(np.int32)
            freq_values = np.asarray([], dtype=np.int64)
        else:
            yt = y_true[:, cols].astype(np.int32)
            yp = y_pred[:, cols].astype(np.int32)
            freq_values = train_freq[cols]
        test_positive_count = int(yt.sum())
        rows.append({
            "segment": segment,
            "segment_label": SEGMENT_DISPLAY[segment],
            "frequency_rule": frequency_rule(segment, head_min_freq, mid_min_freq),
            "label_count": int(len(cols)),
            "train_positive_count": int(freq_values.sum()) if len(freq_values) else 0,
            "test_positive_count": test_positive_count,
            "positive_count": test_positive_count,
            "predicted_count": int(yp.sum()),
            "train_freq_min": int(freq_values.min()) if len(freq_values) else 0,
            "train_freq_median": float(np.median(freq_values)) if len(freq_values) else 0.0,
            "train_freq_max": int(freq_values.max()) if len(freq_values) else 0,
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows
"""


DATASET_LABEL_FREQ_INLINE = r"""
def train_frequency_from_dataset_labels(dataset, train_indices):
    rows = []
    for label_index in train_indices:
        labels = dataset.labels[int(label_index)]
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)
        rows.append(labels.astype(np.int64))
    if not rows:
        return np.zeros(0, dtype=np.int64)
    return np.stack(rows, axis=0).sum(axis=0).astype(np.int64)
"""


HPT_LABEL_FREQ_INLINE = r"""
def train_frequency_from_hpt_json(data_path, num_class):
    counts = np.zeros(int(num_class), dtype=np.int64)
    train_json = Path(data_path) / (Path(data_path).name + "_train.json")
    with train_json.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            labels = json.loads(line).get("label", [])
            for label_id in set(int(label_id) for label_id in labels):
                if 0 <= label_id < int(num_class):
                    counts[label_id] += 1
    return counts
"""


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find inline fragment for replacement: {old[:120]!r}")
    return text.replace(old, new, 1)


def replace_rows_call(inline: str) -> str:
    replacements = [
        (
            '"rows": depth_rows(y_true, y_pred, hd.levels),',
            '"rows": frequency_rows(y_true, y_pred, train_freq, payload["head_min_freq"], payload["mid_min_freq"]),',
        ),
        (
            '"rows": depth_rows(y_true, y_pred, load_levels(data_path)),',
            '"rows": frequency_rows(y_true, y_pred, train_freq, payload["head_min_freq"], payload["mid_min_freq"]),',
        ),
    ]
    for old, new in replacements:
        inline = inline.replace(old, new)
    inline = inline.replace(
        '    "rows": frequency_rows',
        '    "head_min_freq": int(payload["head_min_freq"]),\n'
        '    "mid_min_freq": int(payload["mid_min_freq"]),\n'
        '    "rows": frequency_rows',
    )
    return inline.replace("LABEL_DEPTH_RESULT_JSON", "LABEL_FREQUENCY_RESULT_JSON")


def make_bert_ft_inline() -> str:
    inline = depth.BERT_FT_INLINE
    inline = replace_once(inline, "\n\ndef move_batch", "\n\n" + FREQUENCY_HELPER_INLINE + "\n\ndef move_batch")
    inline = replace_once(
        inline,
        "_, _, _, _, _, test_idx = build_two_stage_split_indices(y_all, cfg)",
        "_, _, _, train_idx, _, test_idx = build_two_stage_split_indices(y_all, cfg)\n"
        "train_freq = y_all[train_idx].sum(axis=0).astype(np.int64)",
    )
    return replace_rows_call(inline)


def make_raehmc_inline() -> str:
    inline = depth.RAEHMC_INLINE
    inline = replace_once(inline, "\n\ncfg = TrainConfig()", "\n\n" + FREQUENCY_HELPER_INLINE + "\n\ncfg = TrainConfig()")
    inline = replace_once(
        inline,
        "train_pool_idx, train_rel_idx, val_rel_idx, _, _, test_idx = build_two_stage_split_indices(y_all, cfg)",
        "train_pool_idx, train_rel_idx, val_rel_idx, train_idx, _, test_idx = build_two_stage_split_indices(y_all, cfg)\n"
        "train_freq = y_all[train_idx].sum(axis=0).astype(np.int64)",
    )
    return replace_rows_call(inline)


def make_hgclr_inline() -> str:
    inline = depth.HGCLR_INLINE
    inline = replace_once(
        inline,
        "    return hd.levels\n\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
        "    return hd.levels\n\n\n" + FREQUENCY_HELPER_INLINE + "\n" + DATASET_LABEL_FREQ_INLINE + "\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
    )
    inline = replace_once(
        inline,
        'split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)',
        'split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)\n'
        'train_freq = train_frequency_from_dataset_labels(dataset, split["train"])',
    )
    return replace_rows_call(inline)


def make_hill_inline() -> str:
    inline = depth.HILL_INLINE
    inline = replace_once(
        inline,
        "    return hd.levels\n\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
        "    return hd.levels\n\n\n" + FREQUENCY_HELPER_INLINE + "\n" + DATASET_LABEL_FREQ_INLINE + "\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
    )
    inline = replace_once(
        inline,
        'split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)',
        'split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)\n'
        'train_freq = train_frequency_from_dataset_labels(dataset, split["train"])',
    )
    return replace_rows_call(inline)


def make_hpt_inline() -> str:
    inline = depth.HPT_INLINE
    inline = replace_once(
        inline,
        "    return hd.levels\n\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
        "    return hd.levels\n\n\n" + FREQUENCY_HELPER_INLINE + "\n" + HPT_LABEL_FREQ_INLINE + "\n\ncheckpoint_path = Path(payload[\"checkpoint\"])",
    )
    inline = replace_once(
        inline,
        "num_class = len(label_dict)",
        "num_class = len(label_dict)\ntrain_freq = train_frequency_from_hpt_json(data_path, num_class)",
    )
    return replace_rows_call(inline)


INLINE_BY_MODEL = {
    "bert_ft": make_bert_ft_inline(),
    "raehmc": make_raehmc_inline(),
    "hgclr": make_hgclr_inline(),
    "hill": make_hill_inline(),
    "hpt": make_hpt_inline(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BERT-FT, RAE-HMC, HGCLR, HILL, and HPT checkpoints by "
            "training-label frequency segments: Head/Mid/Tail."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert_ft", "raehmc", "hgclr", "hill", "hpt"],
        choices=list(depth.MODEL_LABELS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list. Overrides start/end.")
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--dataset-base", default="raehmc_food")
    parser.add_argument("--dataset-csv", default="dataset/dataset.csv", help="Dataset CSV used for RAE-HMC evaluation.")
    parser.add_argument("--hierarchy-json", default="dataset/label_hierarchy.json")
    parser.add_argument("--temp-prefix", default="baselinetmp", help="Used by the default BERT-FT checkpoint template.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plm", default=None, help="Override checkpoint PLM for HGCLR/HILL. Defaults to checkpoint args.")
    parser.add_argument("--max-token", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for HGCLR/HILL sigmoid scores.")
    parser.add_argument("--head-min-freq", type=int, default=100, help="Head labels have train frequency > this value.")
    parser.add_argument("--mid-min-freq", type=int, default=35, help="Mid labels have train frequency >= this value.")
    parser.add_argument("--hgclr-eval-checkpoint", default="_macro", choices=["_macro", "_micro"])
    parser.add_argument("--hill-eval-checkpoint", default="macro", choices=["macro", "micro"])
    parser.add_argument("--hpt-eval-checkpoint", default="_macro", choices=["_macro", "_micro"])
    parser.add_argument("--hpt-temp-prefix", default="hptdepth")
    parser.add_argument("--bert-ft-python", default=depth.default_env_python("RaehmcEnv"))
    parser.add_argument("--raehmc-python", default=depth.default_env_python("RaehmcEnv"))
    parser.add_argument("--hgclr-python", default=depth.default_env_python("hgclr-gpu"))
    parser.add_argument("--hill-python", default=depth.default_env_python("hill-gpu"))
    parser.add_argument("--hpt-python", default=depth.default_env_python("hpt-gpu"))
    parser.add_argument(
        "--bert-ft-checkpoint-template",
        default="baselines/BERT-FT/checkpoints/{temp_prefix}_{dataset_base}_s{seed}/best_model.pt",
    )
    parser.add_argument(
        "--raehmc-checkpoint-template",
        default="outputs/ablation_cl/ablation_cl_runs/seed{seed}/raehmc_cl/best_model_holdout.pt",
    )
    parser.add_argument(
        "--hgclr-checkpoint-template",
        default=(
            "baselines/HGCLR/checkpoints/"
            "{dataset_base}_s{seed}-multiseed_s{seed}/checkpoint_best{hgclr_eval_checkpoint}.pt"
        ),
    )
    parser.add_argument(
        "--hill-checkpoint-template",
        default="baselines/HILL/ckpt/{dataset_base}_s{seed}-multiseed_s{seed}/best_{hill_eval_checkpoint}.pt",
    )
    parser.add_argument(
        "--hpt-checkpoint-template",
        default=(
            "baselines/HPT/checkpoints/"
            "{hpt_temp_prefix}_{dataset_base}_s{seed}-{hpt_temp_prefix}_s{seed}/"
            "checkpoint_best{hpt_eval_checkpoint}.pt"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/baselines_label_frequency")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--y-min", type=float, default=0.8)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--plot-python", default=depth.default_env_python("RaehmcEnv"))
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-model/seed JSON results.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Fail instead of skipping missing checkpoints.")
    parser.add_argument("--no-plot", action="store_true", help="Write CSVs without generating the figure.")
    return parser.parse_args()


def inline_for(model: str) -> str:
    try:
        return INLINE_BY_MODEL[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model}") from exc


def run_model_seed(
    *,
    model: str,
    seed: int,
    args: argparse.Namespace,
    repo_root: Path,
    output_json: Path,
    checkpoint_path: Path,
) -> Dict[str, object]:
    payload = {
        "model": depth.MODEL_LABELS[model],
        "seed": int(seed),
        "checkpoint": str(checkpoint_path),
        "output_json": str(output_json),
        "device": args.device,
        "plm": args.plm,
        "dataset_csv": args.dataset_csv,
        "hierarchy_json": args.hierarchy_json,
        "max_token": int(args.max_token),
        "batch_size": int(args.batch_size),
        "threshold": float(args.threshold),
        "head_min_freq": int(args.head_min_freq),
        "mid_min_freq": int(args.mid_min_freq),
    }
    cmd = [
        depth.python_for(model, args),
        "-c",
        inline_for(model),
        str(repo_root),
        json.dumps(payload, ensure_ascii=False),
    ]
    depth.run_command(cmd, depth.cwd_for(model, repo_root))
    with output_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_result(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in result["rows"]:
        rows.append({
            "model": result["model"],
            "seed": int(result["seed"]),
            "segment": str(row["segment"]),
            "segment_label": str(row["segment_label"]),
            "frequency_rule": str(row["frequency_rule"]),
            "label_count": int(row["label_count"]),
            "train_positive_count": int(row["train_positive_count"]),
            "test_positive_count": int(row["test_positive_count"]),
            "predicted_count": int(row["predicted_count"]),
            "train_freq_min": int(row["train_freq_min"]),
            "train_freq_median": float(row["train_freq_median"]),
            "train_freq_max": int(row["train_freq_max"]),
            "micro_f1": float(row["micro_f1"]),
            "macro_f1": float(row["macro_f1"]),
            "threshold": float(result["threshold"]),
            "checkpoint": result["checkpoint"],
        })
    return rows


def result_matches_frequency_args(result: Dict[str, object], args: argparse.Namespace) -> bool:
    return (
        result.get("head_min_freq") == int(args.head_min_freq)
        and result.get("mid_min_freq") == int(args.mid_min_freq)
    )


def mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    segment_rank = {segment: idx for idx, segment in enumerate(SEGMENT_ORDER)}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["segment"])), []).append(row)

    summary = []
    for (model, segment), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], segment_rank.get(item[0][1], 99)),
    ):
        micro_values = [float(row["micro_f1"]) for row in group]
        macro_values = [float(row["macro_f1"]) for row in group]
        label_counts = [float(row["label_count"]) for row in group]
        train_positive_counts = [float(row["train_positive_count"]) for row in group]
        test_positive_counts = [float(row["test_positive_count"]) for row in group]
        freq_mins = [float(row["train_freq_min"]) for row in group]
        freq_medians = [float(row["train_freq_median"]) for row in group]
        freq_maxes = [float(row["train_freq_max"]) for row in group]
        summary.append({
            "model": model,
            "segment": segment,
            "segment_label": SEGMENT_DISPLAY.get(segment, segment),
            "frequency_rule": group[0]["frequency_rule"],
            "seed_count": len(group),
            "label_count_mean": mean(label_counts),
            "label_count_std": std(label_counts),
            "train_positive_count_mean": mean(train_positive_counts),
            "test_positive_count_mean": mean(test_positive_counts),
            "train_freq_min_mean": mean(freq_mins),
            "train_freq_median_mean": mean(freq_medians),
            "train_freq_max_mean": mean(freq_maxes),
            "micro_f1_mean": mean(micro_values),
            "micro_f1_std": std(micro_values),
            "macro_f1_mean": mean(macro_values),
            "macro_f1_std": std(macro_values),
        })
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(
    summary_rows: List[Dict[str, object]],
    plot_path: Path,
    model_order: List[str],
    y_min: float,
    y_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: Dict[str, List[Dict[str, object]]] = {}
    segment_rank = {segment: idx for idx, segment in enumerate(SEGMENT_ORDER)}
    for row in summary_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    metric_specs = [
        ("micro_f1_mean", "Micro-F1"),
        ("macro_f1_mean", "Macro-F1"),
    ]
    x_positions = list(range(len(SEGMENT_ORDER)))
    for ax, (mean_key, title) in zip(axes, metric_specs):
        for model in model_order:
            model_label = depth.MODEL_LABELS.get(model, model)
            rows = sorted(by_model.get(model_label, []), key=lambda row: segment_rank.get(str(row["segment"]), 99))
            if not rows:
                continue
            xs = [segment_rank[str(row["segment"])] for row in rows]
            means = [float(row[mean_key]) for row in rows]
            ax.plot(xs, means, marker="o", linewidth=2, label=model_label)
        ax.set_title(title)
        ax.set_xlabel("Label frequency segment")
        ax.set_ylabel("F1 score")
        ax.set_ylim(float(y_min), float(y_max))
        ax.set_xticks(x_positions, [SEGMENT_DISPLAY[segment] for segment in SEGMENT_ORDER])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)


def plot_summary_subprocess(
    summary_csv: Path,
    plot_path: Path,
    model_order: List[str],
    plot_python: str,
    y_min: float,
    y_max: float,
) -> None:
    plot_inline = r"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary_csv = Path(sys.argv[1])
plot_path = Path(sys.argv[2])
model_order = json.loads(sys.argv[3])
model_labels = json.loads(sys.argv[4])
y_min = float(sys.argv[5])
y_max = float(sys.argv[6])
segment_order = ["head", "mid", "tail"]
segment_display = {"head": "Head", "mid": "Mid", "tail": "Tail"}
segment_rank = {segment: idx for idx, segment in enumerate(segment_order)}

with summary_csv.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

by_model = {}
for row in rows:
    by_model.setdefault(row["model"], []).append(row)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, (mean_key, title) in zip(axes, [("micro_f1_mean", "Micro-F1"), ("macro_f1_mean", "Macro-F1")]):
    for model in model_order:
        model_label = model_labels.get(model, model)
        model_rows = sorted(by_model.get(model_label, []), key=lambda row: segment_rank.get(row["segment"], 99))
        if not model_rows:
            continue
        xs = [segment_rank[row["segment"]] for row in model_rows]
        means = [float(row[mean_key]) for row in model_rows]
        ax.plot(xs, means, marker="o", linewidth=2, label=model_label)
    ax.set_title(title)
    ax.set_xlabel("Label frequency segment")
    ax.set_ylabel("F1 score")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(list(range(len(segment_order))), [segment_display[s] for s in segment_order])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
axes[-1].legend(loc="best")
fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=220)
plt.close(fig)
"""
    depth.ensure_python_exists(plot_python, "plot")
    cmd = [
        plot_python,
        "-c",
        plot_inline,
        str(summary_csv),
        str(plot_path),
        json.dumps(model_order, ensure_ascii=False),
        json.dumps(depth.MODEL_LABELS, ensure_ascii=False),
        str(float(y_min)),
        str(float(y_max)),
    ]
    depth.run_command(cmd, Path.cwd())


def main() -> None:
    args = parse_args()
    if int(args.mid_min_freq) > int(args.head_min_freq):
        raise ValueError("--mid-min-freq must be <= --head-min-freq.")

    repo_root = Path(__file__).resolve().parent
    seeds = depth.resolve_seeds(args)

    for model in args.models:
        depth.ensure_python_exists(depth.python_for(model, args), depth.MODEL_LABELS[model])

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "per_seed_json"
    result_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "baselines_label_frequency.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "baselines_label_frequency_summary.csv"
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "baselines_label_frequency.png"
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    if not summary_csv.is_absolute():
        summary_csv = repo_root / summary_csv
    if not plot_path.is_absolute():
        plot_path = repo_root / plot_path

    all_rows: List[Dict[str, object]] = []
    missing: List[str] = []

    for seed in seeds:
        print(f"\n[seed {seed}]", flush=True)
        for model in args.models:
            checkpoint_path = depth.checkpoint_for(model, args, repo_root, seed)
            output_json = result_dir / f"{model}_seed{seed}.json"
            if not checkpoint_path.exists():
                message = f"{depth.MODEL_LABELS[model]} seed={seed}: missing checkpoint {checkpoint_path}"
                if args.fail_on_missing:
                    raise FileNotFoundError(message)
                print(f"  - skipping {message}", flush=True)
                missing.append(message)
                continue
            if args.resume and output_json.exists():
                with output_json.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
                if result_matches_frequency_args(result, args):
                    print(f"  - reusing {depth.MODEL_LABELS[model]} result: {output_json}", flush=True)
                else:
                    print(
                        f"  - re-evaluating {depth.MODEL_LABELS[model]}: cached frequency thresholds differ",
                        flush=True,
                    )
                    result = run_model_seed(
                        model=model,
                        seed=seed,
                        args=args,
                        repo_root=repo_root,
                        output_json=output_json,
                        checkpoint_path=checkpoint_path,
                    )
            else:
                print(f"  - evaluating {depth.MODEL_LABELS[model]}: {checkpoint_path}", flush=True)
                result = run_model_seed(
                    model=model,
                    seed=seed,
                    args=args,
                    repo_root=repo_root,
                    output_json=output_json,
                    checkpoint_path=checkpoint_path,
                )
            all_rows.extend(flatten_result(result))

    if not all_rows:
        raise RuntimeError("No label-frequency results were produced. Check checkpoint paths or use --fail-on-missing.")

    row_fields = [
        "model",
        "seed",
        "segment",
        "segment_label",
        "frequency_rule",
        "label_count",
        "train_positive_count",
        "test_positive_count",
        "predicted_count",
        "train_freq_min",
        "train_freq_median",
        "train_freq_max",
        "micro_f1",
        "macro_f1",
        "threshold",
        "checkpoint",
    ]
    write_csv(output_csv, all_rows, row_fields)

    summary_rows = summarize_rows(all_rows)
    summary_fields = [
        "model",
        "segment",
        "segment_label",
        "frequency_rule",
        "seed_count",
        "label_count_mean",
        "label_count_std",
        "train_positive_count_mean",
        "test_positive_count_mean",
        "train_freq_min_mean",
        "train_freq_median_mean",
        "train_freq_max_mean",
        "micro_f1_mean",
        "micro_f1_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    if not args.no_plot:
        try:
            plot_summary(summary_rows, plot_path, args.models, args.y_min, args.y_max)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print(f"Current Python lacks matplotlib; retrying plot with {args.plot_python}", flush=True)
            plot_summary_subprocess(summary_csv, plot_path, args.models, args.plot_python, args.y_min, args.y_max)

    print(f"\nFrequency rules: Head > {int(args.head_min_freq)}, Mid {int(args.mid_min_freq)}-{int(args.head_min_freq)}, Tail < {int(args.mid_min_freq)}")
    print(f"Saved per-seed label-frequency CSV: {output_csv}")
    print(f"Saved seed-averaged summary CSV: {summary_csv}")
    if not args.no_plot:
        print(f"Saved label-frequency plot: {plot_path}")
    if missing:
        print("\nMissing checkpoints skipped:")
        for message in missing:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
