#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_hierarchy_utils import build_multi_hot_Y, load_hierarchy_from_file, parse_label_hierarchy
from encoder import EncoderConfig, SharedEncoder
from train_rae_hmc import (
    TrainConfig,
    bf16_autocast_context,
    build_training_scheduler,
    build_two_stage_split_indices,
    build_weight_decay_param_groups,
    compute_val_metrics,
    micro_f1,
    macro_f1,
    parse_label_cell,
    parse_root_label_names,
    promote_named_roots,
    set_seed,
    strip_root_label,
    tokenize_texts,
    subset_tokens,
)


class BertFtModel(nn.Module):
    def __init__(self, encoder: SharedEncoder, num_labels: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.classifier = nn.Linear(int(encoder.hidden_size), int(num_labels))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        return self.classifier(self.dropout(h))


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a clean BERT fine-tuning baseline with a linear sigmoid head.")
    parser.add_argument("--dataset-csv", default=cfg.dataset_csv)
    parser.add_argument("--hierarchy-json", default=cfg.hierarchy_json)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--pooling", default=cfg.encoder_pooling, choices=["cls", "mean"])
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.classifier_epochs)
    parser.add_argument("--patience", type=int, default=cfg.classifier_patience)
    parser.add_argument("--encoder-lr", type=float, default=cfg.encoder_lr)
    parser.add_argument("--classifier-lr", type=float, default=cfg.classifier_lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--test-ratio", type=float, default=cfg.test_ratio)
    parser.add_argument("--val-ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--val-metric", default=cfg.val_metric, choices=["micro", "macro"])
    parser.add_argument("--delta-candidates", type=float, nargs="+", default=cfg.delta_candidates)
    parser.add_argument("--exclude-root-label", action=argparse.BooleanOptionalAction, default=cfg.exclude_root_label)
    parser.add_argument("--root-label-name", nargs="+", default=parse_root_label_names(cfg.root_label_name))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-bf16-amp", action=argparse.BooleanOptionalAction, default=cfg.use_bf16_amp)
    parser.add_argument("--cache-tokens-on-gpu", action=argparse.BooleanOptionalAction, default=cfg.cache_tokens_on_gpu)
    return parser.parse_args()


def load_hierarchy(args: argparse.Namespace):
    root_names = parse_root_label_names(args.root_label_name)
    if bool(args.exclude_root_label):
        with open(args.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = promote_named_roots(json.load(handle), root_names)
        return parse_label_hierarchy(hjson), root_names
    return load_hierarchy_from_file(args.hierarchy_json), root_names


def make_cfg(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.dataset_csv = str(args.dataset_csv)
    cfg.hierarchy_json = str(args.hierarchy_json)
    cfg.seed = int(args.seed)
    cfg.model_name = str(args.model_name)
    cfg.max_len = int(args.max_len)
    cfg.encoder_pooling = str(args.pooling)
    cfg.batch_size = int(args.batch_size)
    cfg.classifier_epochs = int(args.epochs)
    cfg.classifier_patience = int(args.patience)
    cfg.encoder_lr = float(args.encoder_lr)
    cfg.classifier_lr = float(args.classifier_lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.warmup_ratio = float(args.warmup_ratio)
    cfg.dropout = float(args.dropout)
    cfg.test_ratio = float(args.test_ratio)
    cfg.val_ratio = float(args.val_ratio)
    cfg.val_metric = str(args.val_metric)
    cfg.delta_candidates = [float(v) for v in args.delta_candidates]
    cfg.exclude_root_label = bool(args.exclude_root_label)
    cfg.root_label_name = list(args.root_label_name)
    cfg.cache_tokens_on_gpu = bool(args.cache_tokens_on_gpu)
    cfg.use_bf16_amp = bool(args.use_bf16_amp)
    cfg.workdir = str(args.output_dir)
    return cfg


def move_batch(tokens: Dict[str, torch.Tensor], start: int, end: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value[start:end].to(device) for key, value in tokens.items()}


def select_device(cfg: TrainConfig, requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def tune_delta(
    scores: torch.Tensor,
    y_val: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[float, float, float, float]:
    y_true = (y_val.detach().cpu().numpy() > 0.5).astype(np.int32)
    best_delta = float(cfg.delta)
    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    for delta in list(cfg.delta_candidates):
        y_pred = (scores.detach().cpu().numpy() >= float(delta)).astype(np.int32)
        score, micro, macro = compute_val_metrics(y_true, y_pred, cfg)
        if score > best_score:
            best_delta = float(delta)
            best_score = float(score)
            best_micro = float(micro)
            best_macro = float(macro)
    return best_delta, best_score, best_micro, best_macro


@torch.no_grad()
def predict_scores(
    model: BertFtModel,
    tokens: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    total = tokens["input_ids"].size(0)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = move_batch(tokens, start, end, device)
        logits = model(batch)
        outputs.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(outputs, dim=0)


def train_epoch(
    model: BertFtModel,
    tokens: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    batch_size: int,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.train()
    indices = torch.randperm(labels.size(0))
    total_loss = 0.0
    seen = 0
    for start in range(0, labels.size(0), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = {
            key: value.index_select(0, batch_idx.to(value.device)).to(device)
            for key, value in tokens.items()
        }
        y = labels.index_select(0, batch_idx).to(device)
        optimizer.zero_grad(set_to_none=True)
        with bf16_autocast_context(cfg, device):
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        bs = y.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        seen += bs
    return total_loss / max(1, seen)


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = select_device(cfg, args.device)
    device_str = str(device)
    hd, root_names = load_hierarchy(args)
    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    label_lists = [parse_label_cell(cell) for cell in df_all[cfg.labels_col].tolist()]
    if bool(cfg.exclude_root_label):
        label_lists = [strip_root_label(labels, root_names) for labels in label_lists]
    y_all_np = np.array(build_multi_hot_Y(label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
    train_pool_idx, train_rel_idx, val_rel_idx, _train_abs, _val_abs, test_idx = build_two_stage_split_indices(
        y_all_np, cfg
    )

    df_train_pool = df_all.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_all.iloc[test_idx].reset_index(drop=True)
    y_train_pool = torch.tensor(y_all_np[train_pool_idx], dtype=torch.float32)
    y_train = y_train_pool.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))
    y_val = y_train_pool.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))

    encoder_cfg = EncoderConfig(
        model_name=cfg.model_name,
        max_length=cfg.max_len,
        pooling=cfg.encoder_pooling,
        normalize=False,
        device=device_str,
        amp_enabled=bool(cfg.use_bf16_amp),
        amp_dtype="bf16",
    )
    encoder = SharedEncoder(encoder_cfg)
    model = BertFtModel(encoder, hd.num_labels, cfg.dropout).to(device)

    train_pool_tokens = tokenize_texts(model.encoder.tokenizer, df_train_pool[cfg.text_col].astype(str).tolist(), cfg.max_len)
    train_tokens = subset_tokens(train_pool_tokens, train_rel_idx)
    val_tokens = subset_tokens(train_pool_tokens, val_rel_idx)
    test_tokens = tokenize_texts(model.encoder.tokenizer, df_test[cfg.text_col].astype(str).tolist(), cfg.max_len)

    if bool(cfg.cache_tokens_on_gpu) and device.type == "cuda":
        train_tokens = {k: v.to(device) for k, v in train_tokens.items()}
        val_tokens = {k: v.to(device) for k, v in val_tokens.items()}
        test_tokens = {k: v.to(device) for k, v in test_tokens.items()}

    param_groups = []
    param_groups.extend(build_weight_decay_param_groups(model.encoder.named_parameters(), cfg.encoder_lr, cfg.weight_decay))
    param_groups.extend(build_weight_decay_param_groups(model.classifier.named_parameters(), cfg.classifier_lr, cfg.weight_decay))
    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = max(1, math.ceil(y_train.size(0) / cfg.batch_size))
    scheduler = build_training_scheduler(
        optimizer,
        warmup_steps=int(steps_per_epoch * cfg.classifier_epochs * cfg.warmup_ratio),
        steps_per_epoch=steps_per_epoch,
        cfg=cfg,
    )

    best_score = -1.0
    best_micro = -1.0
    best_macro = -1.0
    best_delta = float(cfg.delta)
    best_epoch = 0
    no_improve = 0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, cfg.classifier_epochs + 1):
        loss = train_epoch(model, train_tokens, y_train, optimizer, scheduler, cfg.batch_size, device, cfg)
        val_scores = predict_scores(model, val_tokens, cfg.batch_size, device)
        delta, score, micro, macro = tune_delta(val_scores, y_val, cfg)
        print(
            f"[Epoch {epoch:03d}] loss={loss:.4f} val_micro={micro:.4f} "
            f"val_macro={macro:.4f} delta={delta:.2f}",
            flush=True,
        )
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_micro = micro
            best_macro = macro
            best_delta = delta
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "num_labels": int(hd.num_labels),
                    "best_epoch": int(best_epoch),
                    "best_delta": float(best_delta),
                    "val_micro": float(best_micro),
                    "val_macro": float(best_macro),
                },
                best_path,
            )
            print(f"  -> saved best checkpoint to {best_path}", flush=True)
        else:
            no_improve += 1
            if no_improve >= cfg.classifier_patience:
                print(f"[Early Stop] no improvement for {cfg.classifier_patience} epochs.", flush=True)
                break

    summary = {
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_delta": best_delta,
        "val_micro": best_micro,
        "val_macro": best_macro,
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"[Train Done] best_epoch={best_epoch} val_micro={best_micro:.4f} "
        f"val_macro={best_macro:.4f} delta={best_delta:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
