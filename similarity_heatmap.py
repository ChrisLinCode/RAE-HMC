"""
Plot seed-averaged cosine-similarity heatmaps for CL ablation checkpoints.

Usage (from repo root):
  python similarity_heatmap.py --seed-start 41 --seed-end 70
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

from encoder import SharedEncoder, EncoderConfig


SCENARIO_SPECS = [
    ("raehmc_no_cl", "RAE-HMC (w/o CL)"),
    ("raehmc_ss_cl", r"RAE-HMC (w/ $L_{\mathrm{SS}}$)"),
    ("raehmc_hnm_cl", r"RAE-HMC (w/ $L_{\mathrm{HNM}}$)"),
    ("raehmc_cl", r"RAE-HMC (w/ $L_{\mathrm{SS}} + L_{\mathrm{HNM}}$)"),
]


def parse_args():
    # 若不想每次下參數，可直接修改 DEFAULT_*，不帶參數執行即可
    DEFAULT_MODEL = "bert-base-chinese"
    DEFAULT_CHECKPOINT_ROOT = "outputs/ablation_cl/ablation_cl_runs"
    DEFAULT_MAX_LEN = 32
    DEFAULT_POOLING = "mean"
    DEFAULT_RAW_VMIN = -0.2
    DEFAULT_RAW_VMAX = 0.8
    DEFAULT_DIFF_VMIN = -0.2
    DEFAULT_DIFF_VMAX = 0.2
    DEFAULT_OUT = "outputs/similarity_heatmap/cl_similarity_heatmaps_raw_avg.png"

    ap = argparse.ArgumentParser(
        description=(
            "Average text embedding similarity matrices across seed checkpoints "
            "for CL ablation scenarios."
        )
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="encoder backbone model")
    ap.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seed-start", type=int, default=41)
    ap.add_argument("--seed-end", type=int, default=70)
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=[key for key, _ in SCENARIO_SPECS],
        choices=[key for key, _ in SCENARIO_SPECS],
    )
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--pooling", choices=["mean", "cls"], default=DEFAULT_POOLING)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--include-pretrained", action="store_true", help="also show the pretrained encoder before fine-tuning")
    ap.add_argument("--plot-mode", choices=["raw", "diff", "both"], default="raw")
    ap.add_argument("--raw-cmap", default="coolwarm")
    ap.add_argument("--diff-cmap", default="coolwarm")
    ap.add_argument("--raw-vmin", type=float, default=DEFAULT_RAW_VMIN)
    ap.add_argument("--raw-vmax", type=float, default=DEFAULT_RAW_VMAX)
    ap.add_argument("--diff-vmin", type=float, default=DEFAULT_DIFF_VMIN)
    ap.add_argument("--diff-vmax", type=float, default=DEFAULT_DIFF_VMAX)
    ap.add_argument(
        "--mask-diagonal",
        action="store_true",
        help="Hide diagonal self-similarity cells so off-diagonal structure is easier to read.",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="output image path")
    return ap.parse_args()


def load_texts():
    return [
        "特選美式咖啡",
        "英曙格雷紅茶",
        "台灣上等蕉(裸賣)",
        "美國紅寶石無籽葡萄袋",
        "蜜汁叉燒包",
        "韓國麵包三兄弟",
    ]


def configure_chinese_font():
    """Ensure Matplotlib knows a CJK font even if the cache misses it."""
    preferred_names = [
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Noto Serif CJK TC",
        "AR PL UMing TW MBE",
    ]
    for name in preferred_names:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = [name]
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
        except ValueError:
            continue
    else:
        candidate_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        for path in candidate_paths:
            if Path(path).exists():
                font_manager.fontManager.addfont(path)
                prop = font_manager.FontProperties(fname=path)
                font_name = prop.get_name()
                matplotlib.rcParams["font.family"] = [font_name]
                matplotlib.rcParams["font.sans-serif"] = [font_name]
                print(f"[Info] Registering CJK font from {path}")
                break
        else:
            print("[Warn] 找不到可用的 CJK 字型，圖表可能無法顯示中文")
    matplotlib.rcParams["axes.unicode_minus"] = False


def resolve_seeds(args):
    if args.seeds:
        return sorted(dict.fromkeys(int(seed) for seed in args.seeds))
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def scenario_display_name(scenario_key):
    return dict(SCENARIO_SPECS).get(scenario_key, scenario_key)


def checkpoint_path(checkpoint_root, seed, scenario_key):
    return Path(checkpoint_root) / f"seed{int(seed)}" / scenario_key / "best_model_holdout.pt"


def similarity_matrix(enc, texts, batch_size):
    embeddings = enc.encode_texts(texts, batch_size=batch_size).cpu().numpy()
    return np.matmul(embeddings, embeddings.T)


def load_encoder_state(enc, ckpt_path, device):
    # Local experiment checkpoints include config objects, so PyTorch 2.6+ needs weights_only=False.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "encoder_state" not in ckpt:
        raise RuntimeError(f"{ckpt_path} does not contain encoder_state.")
    enc.load_state_dict(ckpt["encoder_state"])
    enc.to(device)
    del ckpt


def average_similarity_for_scenario(enc, texts, seeds, scenario_key, args, device):
    matrix_sum = None
    count = 0
    for seed in seeds:
        ckpt_path = checkpoint_path(args.checkpoint_root, seed, scenario_key)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_encoder_state(enc, ckpt_path, device)
        matrix = similarity_matrix(enc, texts, args.batch_size)
        matrix_sum = matrix if matrix_sum is None else matrix_sum + matrix
        count += 1
        print(f"[{scenario_key}] seed={seed} loaded: {ckpt_path}")
    if count == 0:
        raise ValueError(f"No checkpoints were averaged for scenario: {scenario_key}")
    return matrix_sum / float(count), count


def output_path_for_mode(out_path: Path, mode: str) -> Path:
    if mode == "raw":
        return out_path.with_name(f"{out_path.stem}_raw{out_path.suffix}")
    if mode == "diff":
        return out_path.with_name(f"{out_path.stem}_diff{out_path.suffix}")
    raise ValueError(f"Unsupported output mode: {mode}")


def plot_heatmap_grid(
    panels,
    text_labels,
    out_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    mask_diagonal: bool = False,
) -> None:
    if vmax <= vmin:
        raise ValueError("Heatmap vmax must be greater than vmin.")
    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("No panels to plot.")
    ncols = 3 if n_panels == 3 else (2 if n_panels <= 4 else 3)
    nrows = math.ceil(n_panels / ncols)
    fig = plt.figure(figsize=(5.9 * ncols + 0.45, 6.25 * nrows))
    grid = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.045],
        left=0.07,
        right=0.965,
        bottom=0.20,
        top=0.84,
        wspace=0.18,
        hspace=0.58,
    )
    axes_flat = [fig.add_subplot(grid[row, col]) for row in range(nrows) for col in range(ncols)]
    cbar_ax = fig.add_subplot(grid[:, -1])
    heatmap_kwargs = {
        "cmap": cmap,
        "square": True,
        "xticklabels": text_labels,
        "yticklabels": text_labels,
        "annot": True,
        "annot_kws": {"size": 10},
        "fmt": ".2f",
        "vmin": vmin,
        "vmax": vmax,
        "center": 0.0 if vmin < 0.0 < vmax else None,
    }
    diag_mask = np.eye(len(text_labels), dtype=bool) if mask_diagonal else None

    for idx, (title, matrix, subtitle) in enumerate(panels):
        ax = axes_flat[idx]
        sns.heatmap(
            matrix,
            ax=ax,
            cbar=(idx == n_panels - 1),
            cbar_ax=cbar_ax if idx == n_panels - 1 else None,
            mask=diag_mask,
            **heatmap_kwargs,
        )
        ax.set_title(f"{title}\n({subtitle})" if subtitle else title, fontsize=13, pad=14)
        row_idx = idx // ncols
        is_bottom_row = row_idx == nrows - 1
        if is_bottom_row:
            ax.tick_params(axis="x", labelrotation=45, labelsize=10, pad=3)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        if idx % ncols == 0:
            ax.tick_params(axis="y", labelrotation=0, labelsize=10)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    cbar_ax.tick_params(labelsize=10)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    if args.raw_vmax <= args.raw_vmin:
        raise ValueError("--raw-vmax must be greater than --raw-vmin.")
    if args.diff_vmax <= args.diff_vmin:
        raise ValueError("--diff-vmax must be greater than --diff-vmin.")
    if not args.scenarios and not args.include_pretrained:
        raise ValueError("No panels to plot. Provide --scenarios or --include-pretrained.")
    seeds = resolve_seeds(args)
    texts = load_texts()
    configure_chinese_font()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SharedEncoder(
        EncoderConfig(
            model_name=args.model,
            max_length=args.max_len,
            pooling=args.pooling,
            normalize=True,
            device=device,
        )
    )
    pretrained_matrix = None
    if args.include_pretrained and args.plot_mode in {"raw", "both"}:
        pretrained_matrix = similarity_matrix(enc, texts, args.batch_size)

    scenario_results = {}
    for scenario_key in args.scenarios:
        matrix, count = average_similarity_for_scenario(enc, texts, seeds, scenario_key, args, device)
        scenario_results[scenario_key] = (matrix, count)

    text_labels = np.array(texts)
    out_path = Path(args.out)
    written_paths = []

    if args.plot_mode in {"raw", "both"}:
        raw_panels = []
        if args.include_pretrained:
            raw_panels.append(("Pretrained encoder", pretrained_matrix, "single model"))
        for scenario_key in args.scenarios:
            matrix, count = scenario_results[scenario_key]
            subtitle = f"mean over {count} seeds" if count > 1 else "single model"
            raw_panels.append((scenario_display_name(scenario_key), matrix, subtitle))
        raw_out = out_path if args.plot_mode == "raw" else output_path_for_mode(out_path, "raw")
        plot_heatmap_grid(
            raw_panels,
            text_labels,
            raw_out,
            cmap=args.raw_cmap,
            vmin=args.raw_vmin,
            vmax=args.raw_vmax,
            mask_diagonal=args.mask_diagonal,
        )
        written_paths.append(raw_out)

    if args.plot_mode in {"diff", "both"}:
        if "raehmc_no_cl" not in scenario_results:
            matrix, count = average_similarity_for_scenario(enc, texts, seeds, "raehmc_no_cl", args, device)
            scenario_results["raehmc_no_cl"] = (matrix, count)
        baseline, count = scenario_results["raehmc_no_cl"]
        diff_panels = []
        for scenario_key in args.scenarios:
            if scenario_key == "raehmc_no_cl":
                continue
            matrix, scenario_count = scenario_results[scenario_key]
            diff_panels.append((
                f"{scenario_display_name(scenario_key)} - w/o CL",
                matrix - baseline,
                f"mean over {scenario_count} seeds",
            ))
        if not diff_panels:
            raise ValueError("Diff mode requires at least one scenario other than raehmc_no_cl.")
        diff_out = out_path if args.plot_mode == "diff" else output_path_for_mode(out_path, "diff")
        plot_heatmap_grid(
            diff_panels,
            text_labels,
            diff_out,
            cmap=args.diff_cmap,
            vmin=args.diff_vmin,
            vmax=args.diff_vmax,
            mask_diagonal=args.mask_diagonal,
        )
        written_paths.append(diff_out)

    for path in written_paths:
        print(f"[OK] Saved heatmap to {path} with {text_labels.shape[0]} texts and {len(seeds)} seeds")


if __name__ == "__main__":
    main()
