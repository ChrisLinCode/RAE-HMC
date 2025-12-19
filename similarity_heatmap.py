"""
Plot a cosine-similarity heatmap of app embeddings to inspect separability.

Usage (from repo root):
  python similarity_heatmap.py --dataset dataset/dataset.csv --model bert-base-chinese --max_len 32 --sample 300 --min_count 5 --out heatmap.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

from encoder import SharedEncoder, EncoderConfig


def parse_args():
    # 若不想每次下參數，可直接修改 DEFAULT_*，不帶參數執行即可
    DEFAULT_MODEL_PRE = "bert-base-chinese"
    DEFAULT_MODEL_POST = "bert-base-chinese"  # 用相同架構，權重可由 ckpt 覆寫
    DEFAULT_CKPT = "./outputs/best_model_full.pt"
    DEFAULT_MAX_LEN = 32
    DEFAULT_OUT = "heatmap.png"

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pre", default=DEFAULT_MODEL_PRE, help="pretrained model (訓練前)")
    ap.add_argument("--model_post", default=DEFAULT_MODEL_POST, help="finetuned model (訓練後)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help=".pt checkpoint with encoder_state; 如果提供則載入此權重")
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
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


def main():
    args = parse_args()
    texts = load_texts()
    configure_chinese_font()
    cmap = LinearSegmentedColormap.from_list("bright_red_white", ["#ffffff","#ff0033"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Encode with pre- and post-training models
    enc_pre = SharedEncoder(EncoderConfig(model_name=args.model_pre, max_length=args.max_len, pooling="cls", normalize=True, device=device))
    enc_post = SharedEncoder(EncoderConfig(model_name=args.model_post, max_length=args.max_len, pooling="cls", normalize=True, device=device))
    # 若提供 ckpt，覆寫 post 權重
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        if "encoder_state" in ckpt:
            enc_post.load_state_dict(ckpt["encoder_state"], strict=False)
            print(f"[Info] Loaded encoder_state from {args.ckpt}")
        else:
            print(f"[Warn] {args.ckpt} 不包含 encoder_state，改用 {args.model_post} 的預設權重")

    X_pre = enc_pre.encode_texts(texts, batch_size=64).cpu().numpy()
    X_post = enc_post.encode_texts(texts, batch_size=64).cpu().numpy()
    texts = np.array(texts)

    S_pre = np.matmul(X_pre, X_pre.T)   # cosine similarity if normalized
    S_post = np.matmul(X_post, X_post.T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(S_pre, cmap=cmap, square=True, xticklabels=texts, yticklabels=texts,
                annot=True, fmt=".2f", cbar=True, ax=axes[0])
    axes[0].set_title("訓練前相似度")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(S_post, cmap=cmap, square=True, xticklabels=texts, yticklabels=texts,
                annot=True, fmt=".2f", cbar=True, ax=axes[1])
    axes[1].set_title("訓練後相似度")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Saved heatmap to {out_path} with {texts.shape[0]} samples")


if __name__ == "__main__":
    main()
