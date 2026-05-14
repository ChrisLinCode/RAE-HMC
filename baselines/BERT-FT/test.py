#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from build_hierarchy_utils import build_multi_hot_Y, load_hierarchy_from_file, parse_label_hierarchy
from encoder import EncoderConfig, SharedEncoder
from train_rae_hmc import (
    TrainConfig,
    build_two_stage_split_indices,
    micro_f1,
    macro_f1,
    parse_label_cell,
    parse_root_label_names,
    promote_named_roots,
    set_seed,
    strip_root_label,
    tokenize_texts,
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
    parser = argparse.ArgumentParser(description="Evaluate a BERT-FT baseline checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def cfg_from_payload(payload: dict) -> TrainConfig:
    cfg = TrainConfig()
    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_hierarchy(cfg: TrainConfig):
    root_names = parse_root_label_names(cfg.root_label_name)
    if bool(cfg.exclude_root_label):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = promote_named_roots(json.load(handle), root_names)
        return parse_label_hierarchy(hjson), root_names
    return load_hierarchy_from_file(cfg.hierarchy_json), root_names


def select_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def move_batch(tokens: Dict[str, torch.Tensor], start: int, end: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value[start:end].to(device) for key, value in tokens.items()}


@torch.no_grad()
def predict_scores(model: BertFtModel, tokens: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    total = tokens["input_ids"].size(0)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = move_batch(tokens, start, end, device)
        logits = model(batch)
        outputs.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_payload(checkpoint["config"])
    set_seed(int(cfg.seed))
    device = select_device(args.device)
    device_str = str(device)

    hd, root_names = load_hierarchy(cfg)
    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    label_lists = [parse_label_cell(cell) for cell in df_all[cfg.labels_col].tolist()]
    if bool(cfg.exclude_root_label):
        label_lists = [strip_root_label(labels, root_names) for labels in label_lists]
    y_all_np = np.array(build_multi_hot_Y(label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
    _train_pool_idx, _train_rel_idx, _val_rel_idx, _train_abs, _val_abs, test_idx = build_two_stage_split_indices(
        y_all_np, cfg
    )
    df_test = df_all.iloc[test_idx].reset_index(drop=True)
    y_test = y_all_np[test_idx]

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
    model = BertFtModel(encoder, int(checkpoint["num_labels"]), float(cfg.dropout)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokens = tokenize_texts(model.encoder.tokenizer, df_test[cfg.text_col].astype(str).tolist(), cfg.max_len)
    scores = predict_scores(model, tokens, int(cfg.batch_size), device)
    delta = float(checkpoint["best_delta"])
    y_pred = (scores.numpy() >= delta).astype(np.int32)
    y_true = (y_test > 0.5).astype(np.int32)
    micro = micro_f1(y_true, y_pred)
    macro = macro_f1(y_true, y_pred)

    payload = {
        "checkpoint": str(checkpoint_path),
        "delta": delta,
        "micro": float(micro),
        "macro": float(macro),
    }
    output_json = args.output_json
    if output_json is None:
        output_json = str(checkpoint_path.parent / "test_summary.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"micro-f1: {micro:.4f} macro-f1: {macro:.4f}", flush=True)
    print("BERT_FT_RESULT_JSON=" + json.dumps({"micro": micro, "macro": macro}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
