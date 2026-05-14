#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple


PARAM_ORDER = ["rho", "top_b", "eta"]
PARAM_TITLES = {
    "rho": r"(a) $\rho$ sensitivity",
    "top_b": r"(b) $k_{\ell}$ sensitivity",
    "eta": r"(c) $\eta$ sensitivity",
}
PARAM_XLABELS = {
    "rho": r"$\rho$",
    "top_b": r"$k_{\ell}$",
    "eta": r"$\eta$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot post-hoc memory sensitivity curves.")
    parser.add_argument("--input-csv", default="outputs/sensitivity_memory/sensitivity_memory_hnm.csv")
    parser.add_argument("--output", default="outputs/sensitivity_memory/sensitivity_memory_hnm_plot.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF copy next to the requested output image.",
    )
    return parser.parse_args()


def load_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "param_name": raw["param_name"],
                    "param_value": float(raw["param_value"]),
                    "micro": float(raw["micro"]),
                    "macro": float(raw["macro"]),
                }
            )
    return rows


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, List[Tuple[float, float, float]]]:
    grouped: Dict[Tuple[str, float], Dict[str, List[float]]] = defaultdict(lambda: {"micro": [], "macro": []})
    for row in rows:
        key = (str(row["param_name"]), float(row["param_value"]))
        grouped[key]["micro"].append(float(row["micro"]))
        grouped[key]["macro"].append(float(row["macro"]))

    result: Dict[str, List[Tuple[float, float, float]]] = {}
    for param_name in PARAM_ORDER:
        values: List[Tuple[float, float, float]] = []
        for (name, value), metrics in grouped.items():
            if name != param_name:
                continue
            values.append((value, mean(metrics["micro"]), mean(metrics["macro"])))
        result[param_name] = sorted(values, key=lambda item: item[0])
    return result


def set_adaptive_ylim(ax, micro_values: List[float], macro_values: List[float]) -> None:
    all_values = micro_values + macro_values
    ymin = min(all_values)
    ymax = max(all_values)
    span = max(ymax - ymin, 0.002)
    margin = max(span * 0.22, 0.0015)
    ax.set_ylim(max(0.0, ymin - margin), min(1.0, ymax + margin))


def plot(input_csv: str, output: str, dpi: int, write_pdf: bool) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(input_csv)
    aggregated = aggregate(rows)
    missing = [name for name in PARAM_ORDER if not aggregated.get(name)]
    if missing:
        raise ValueError(f"Missing sensitivity rows for parameter(s): {', '.join(missing)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    colors = {"macro": "#1f5a9d", "micro": "#d08324"}

    for ax, param_name in zip(axes, PARAM_ORDER):
        data = aggregated[param_name]
        x_values = [item[0] for item in data]
        micro_values = [item[1] for item in data]
        macro_values = [item[2] for item in data]

        ax.plot(
            x_values,
            macro_values,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            color=colors["macro"],
            label="Macro-F1",
        )
        ax.plot(
            x_values,
            micro_values,
            marker="s",
            linewidth=2.0,
            markersize=4.5,
            color=colors["micro"],
            label="Micro-F1",
        )
        ax.set_title(PARAM_TITLES[param_name], pad=8)
        ax.set_xlabel(PARAM_XLABELS[param_name])
        ax.set_ylabel("F1")
        ax.set_xticks(x_values)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        set_adaptive_ylim(ax, micro_values, macro_values)

    axes[0].legend(loc="lower right", frameon=False)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if write_pdf:
        pdf_path = os.path.splitext(output)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.input_csv, args.output, args.dpi, write_pdf=args.pdf)
    print(f"[Save] Sensitivity plot saved to {args.output}")
    if args.pdf:
        print(f"[Save] PDF copy saved to {os.path.splitext(args.output)[0] + '.pdf'}")


if __name__ == "__main__":
    main()
