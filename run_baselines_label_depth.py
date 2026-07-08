#!/usr/bin/env python3
"""Evaluate baseline checkpoints by hierarchy label depth and plot F1 curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


BERT_FT_INLINE = r"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

repo_root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
sys.path.insert(0, str(repo_root))

from build_hierarchy_utils import build_multi_hot_Y, load_hierarchy_from_file, parse_label_hierarchy
from encoder import EncoderConfig, SharedEncoder
from train_rae_hmc import (
    TrainConfig,
    build_two_stage_split_indices,
    parse_label_cell,
    parse_root_label_names,
    promote_named_roots,
    set_seed,
    strip_root_label,
    tokenize_texts,
)


class BertFtModel(nn.Module):
    def __init__(self, encoder, num_labels, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.classifier = nn.Linear(int(encoder.hidden_size), int(num_labels))

    def forward(self, batch):
        h = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        return self.classifier(self.dropout(h))


def cfg_from_payload(payload_dict):
    cfg = TrainConfig()
    for key, value in payload_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def select_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_hierarchy(cfg):
    root_names = parse_root_label_names(cfg.root_label_name)
    if bool(cfg.exclude_root_label):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = promote_named_roots(json.load(handle), root_names)
        return parse_label_hierarchy(hjson), root_names
    return load_hierarchy_from_file(cfg.hierarchy_json), root_names


def micro_f1_np(y_true, y_pred):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


def macro_f1_np(y_true, y_pred):
    if y_true.shape[1] == 0:
        return 0.0
    scores = []
    yt_all = y_true.astype(bool)
    yp_all = y_pred.astype(bool)
    for col in range(y_true.shape[1]):
        yt = yt_all[:, col]
        yp = yp_all[:, col]
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def depth_rows(y_true, y_pred, levels):
    labels_by_level = defaultdict(list)
    for label_id, level in levels.items():
        labels_by_level[int(level)].append(int(label_id))
    rows = []
    for level in sorted(labels_by_level):
        cols = np.array(sorted(labels_by_level[level]), dtype=int)
        yt = y_true[:, cols].astype(np.int32)
        yp = y_pred[:, cols].astype(np.int32)
        rows.append({
            "level": int(level),
            "label_count": int(len(cols)),
            "positive_count": int(yt.sum()),
            "predicted_count": int(yp.sum()),
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows


def move_batch(tokens, start, end, device):
    return {key: value[start:end].to(device) for key, value in tokens.items()}


@torch.no_grad()
def predict_scores(model, tokens, batch_size, device):
    model.eval()
    outputs = []
    total = tokens["input_ids"].size(0)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        outputs.append(torch.sigmoid(model(move_batch(tokens, start, end, device))).detach().cpu())
    return torch.cat(outputs, dim=0)


checkpoint_path = Path(payload["checkpoint"])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
cfg = cfg_from_payload(checkpoint["config"])
set_seed(int(cfg.seed))
device = select_device(payload["device"])

hd, root_names = load_hierarchy(cfg)
df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
label_lists = [parse_label_cell(cell) for cell in df_all[cfg.labels_col].tolist()]
if bool(cfg.exclude_root_label):
    label_lists = [strip_root_label(labels, root_names) for labels in label_lists]
y_all = np.array(build_multi_hot_Y(label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
_, _, _, _, _, test_idx = build_two_stage_split_indices(y_all, cfg)
df_test = df_all.iloc[test_idx].reset_index(drop=True)
y_true = y_all[test_idx].astype(np.int32)

encoder_cfg = EncoderConfig(
    model_name=cfg.model_name,
    max_length=cfg.max_len,
    pooling=cfg.encoder_pooling,
    normalize=False,
    device=str(device),
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

result = {
    "model": payload["model"],
    "seed": int(payload["seed"]),
    "checkpoint": str(checkpoint_path),
    "threshold": delta,
    "rows": depth_rows(y_true, y_pred, hd.levels),
}
with open(payload["output_json"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print("LABEL_DEPTH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
"""


RAEHMC_INLINE = r"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

repo_root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
sys.path.insert(0, str(repo_root))

from build_hierarchy_utils import build_multi_hot_Y, load_hierarchy_from_file, make_level_slices, parse_label_hierarchy
from inference import Hierarchy
from memory import MemoryConfig
from train_rae_hmc import (
    TrainConfig,
    build_label_descriptions,
    build_two_stage_split_indices,
    evaluate_model_on_test_split,
    load_model_from_checkpoint_for_test,
    normalize_retrieval_protocol,
    parse_label_cell,
    parse_root_label_names,
    promote_named_roots,
    set_seed,
    strip_root_label,
    subset_tokens,
    tokenize_texts,
    tune_validation_strategy,
)


def select_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def load_hierarchy(cfg):
    root_names = parse_root_label_names(cfg.root_label_name)
    if bool(cfg.exclude_root_label):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as handle:
            hjson = promote_named_roots(json.load(handle), root_names)
        return parse_label_hierarchy(hjson), root_names
    return load_hierarchy_from_file(cfg.hierarchy_json), root_names


def micro_f1_np(y_true, y_pred):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


def macro_f1_np(y_true, y_pred):
    if y_true.shape[1] == 0:
        return 0.0
    scores = []
    yt_all = y_true.astype(bool)
    yp_all = y_pred.astype(bool)
    for col in range(y_true.shape[1]):
        yt = yt_all[:, col]
        yp = yp_all[:, col]
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def depth_rows(y_true, y_pred, levels):
    labels_by_level = defaultdict(list)
    for label_id, level in levels.items():
        labels_by_level[int(level)].append(int(label_id))
    rows = []
    for level in sorted(labels_by_level):
        cols = np.array(sorted(labels_by_level[level]), dtype=int)
        yt = y_true[:, cols].astype(np.int32)
        yp = y_pred[:, cols].astype(np.int32)
        rows.append({
            "level": int(level),
            "label_count": int(len(cols)),
            "positive_count": int(yt.sum()),
            "predicted_count": int(yp.sum()),
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows


cfg = TrainConfig()
cfg.seed = int(payload["seed"])
cfg.dataset_csv = str(resolve_path(payload["dataset_csv"]))
cfg.hierarchy_json = str(resolve_path(payload["hierarchy_json"]))
cfg.workdir = str(Path(payload["output_json"]).parent)
cfg.print_per_label_metrics = False
set_seed(int(cfg.seed))
device = select_device(payload["device"])
device_str = str(device)

hd, root_names = load_hierarchy(cfg)
level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
label_levels = [level_lookup.get(i, 1) for i in range(hd.num_labels)]

checkpoint_path = Path(payload["checkpoint"])
checkpoint_probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
checkpoint_level_sizes = checkpoint_probe.get("clf_cfg", {}).get("level_sizes")
if checkpoint_level_sizes is not None and list(checkpoint_level_sizes) != list(hd.level_sizes):
    raise RuntimeError(
        "RAE-HMC checkpoint label depth sizes do not match the requested hierarchy: "
        f"checkpoint={checkpoint_level_sizes}, hierarchy={hd.level_sizes}. "
        "Use --raehmc-checkpoint-template / --dataset-csv / --hierarchy-json for matching artifacts."
    )
del checkpoint_probe

df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
label_lists = [parse_label_cell(cell) for cell in df_all[cfg.labels_col].tolist()]
if bool(cfg.exclude_root_label):
    label_lists = [strip_root_label(labels, root_names) for labels in label_lists]
y_all = np.array(build_multi_hot_Y(label_lists, hd.label2id, hd.ancestors, add_ancestors=True))
train_pool_idx, train_rel_idx, val_rel_idx, _, _, test_idx = build_two_stage_split_indices(y_all, cfg)
df_train = df_all.iloc[train_pool_idx].reset_index(drop=True)
df_test = df_all.iloc[test_idx].reset_index(drop=True)
y_train_full = torch.tensor(y_all[train_pool_idx], dtype=torch.float32)
y_test = torch.tensor(y_all[test_idx], dtype=torch.float32)

enc, clf, checkpoint = load_model_from_checkpoint_for_test(
    cfg=cfg,
    checkpoint_path=str(checkpoint_path),
    device=device,
    device_str=device_str,
)
label_descs = build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
train_tokens = tokenize_texts(enc.tokenizer, df_train[cfg.text_col].astype(str).tolist(), cfg.max_len)
test_tokens = tokenize_texts(enc.tokenizer, df_test[cfg.text_col].astype(str).tolist(), cfg.max_len)
label_tokens = tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
eval_train_tokens = subset_tokens(train_tokens, train_rel_idx)
eval_train_labels = y_train_full.index_select(0, torch.tensor(train_rel_idx, dtype=torch.long))

eta_final = float(checkpoint.get("eta", cfg.eta))
delta_final = float(checkpoint.get("delta", cfg.delta))
rho_final = float(checkpoint.get("rho", cfg.rho))
top_b_final = int(checkpoint.get("top_b", cfg.top_b))
top_b_levels_final = checkpoint.get("top_b_levels", None)
delta_levels_final = checkpoint.get("delta_levels", None)
mem_cfg = MemoryConfig(
    backend="faiss_ip",
    top_b=top_b_final,
    tau_mem=cfg.tau_mem,
    rho=rho_final,
    device=device_str,
)
if bool(cfg.use_memory) and normalize_retrieval_protocol(cfg) == "post_hoc":
    val_tokens_for_tuning = subset_tokens(train_tokens, val_rel_idx)
    val_labels_for_tuning = y_train_full.index_select(0, torch.tensor(val_rel_idx, dtype=torch.long))
    posthoc_result = tune_validation_strategy(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=Hierarchy(num_labels=hd.num_labels, ancestors=hd.ancestors),
        level_slices=make_level_slices(hd.levels),
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
    rho_final = float(posthoc_result["rho"])
    top_b_final = int(posthoc_result["top_b"])
    top_b_levels_final = posthoc_result.get("top_b_levels", None)
    delta_levels_final = posthoc_result.get("delta_levels", None)
    mem_cfg = MemoryConfig(
        backend="faiss_ip",
        top_b=top_b_final,
        tau_mem=cfg.tau_mem,
        rho=rho_final,
        device=device_str,
    )
result_payload = evaluate_model_on_test_split(
    cfg=cfg,
    hd=hd,
    hierarchy_obj=Hierarchy(num_labels=hd.num_labels, ancestors=hd.ancestors),
    label_levels=label_levels,
    label_tokens=label_tokens,
    train_tokens_for_memory=eval_train_tokens,
    Y_train_for_memory=eval_train_labels,
    test_tokens=test_tokens,
    Y_te=y_test,
    enc=enc,
    clf=clf,
    device=device,
    device_str=device_str,
    mem_cfg=mem_cfg,
    eta_final=eta_final,
    delta_final=delta_final,
    top_b_final=top_b_final,
    top_b_levels_final=top_b_levels_final,
    delta_levels_final=delta_levels_final,
)
y_true = result_payload["y_true_te"].astype(np.int32)
y_pred = result_payload["y_pred_te"].astype(np.int32)

result = {
    "model": payload["model"],
    "seed": int(payload["seed"]),
    "checkpoint": str(checkpoint_path),
    "threshold": delta_final,
    "rows": depth_rows(y_true, y_pred, hd.levels),
}
with open(payload["output_json"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print("LABEL_DEPTH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
"""


HGCLR_INLINE = r"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

repo_root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path.cwd()))

from build_hierarchy_utils import parse_label_hierarchy
from data.prepare_raehmc import parse_root_label_names, promote_named_roots
from model.contrast import ContrastModel
from train import BertDataset, resolve_max_token


def select_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def micro_f1_np(y_true, y_pred):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


def macro_f1_np(y_true, y_pred):
    if y_true.shape[1] == 0:
        return 0.0
    scores = []
    yt_all = y_true.astype(bool)
    yp_all = y_pred.astype(bool)
    for col in range(y_true.shape[1]):
        yt = yt_all[:, col]
        yp = yp_all[:, col]
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def depth_rows(y_true, y_pred, levels):
    labels_by_level = defaultdict(list)
    for label_id, level in levels.items():
        labels_by_level[int(level)].append(int(label_id))
    rows = []
    for level in sorted(labels_by_level):
        cols = np.array(sorted(labels_by_level[level]), dtype=int)
        yt = y_true[:, cols].astype(np.int32)
        yp = y_pred[:, cols].astype(np.int32)
        rows.append({
            "level": int(level),
            "label_count": int(len(cols)),
            "positive_count": int(yt.sum()),
            "predicted_count": int(yp.sum()),
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows


def load_levels(data_path):
    meta_path = Path(data_path) / "dataset_meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(meta["hierarchy_json"], "r", encoding="utf-8") as handle:
        hierarchy_payload = json.load(handle)
    root_names = parse_root_label_names(meta.get("root_label_name", []))
    if bool(meta.get("exclude_root_label", True)):
        hierarchy_payload = promote_named_roots(hierarchy_payload, root_names)
    hd = parse_label_hierarchy(hierarchy_payload)
    return hd.levels


checkpoint_path = Path(payload["checkpoint"])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
run_args = checkpoint["args"]
device = select_device(payload["device"])
data_path = os.path.join("data", run_args.data)
plm_name = payload["plm"] or getattr(run_args, "plm", "bert-base-uncased")
max_token = resolve_max_token(data_path, payload["max_token"])
threshold = float(payload["threshold"])

tokenizer = AutoTokenizer.from_pretrained(plm_name)
label_dict = torch.load(os.path.join(data_path, "bert_value_dict.pt"), map_location="cpu", weights_only=False)
num_class = len(label_dict)
dataset = BertDataset(max_token=max_token, device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)
test_loader = DataLoader(
    Subset(dataset, split["test"]),
    batch_size=int(payload["batch_size"]),
    shuffle=False,
    collate_fn=dataset.collate_fn,
)
if not hasattr(run_args, "graph"):
    run_args.graph = False
model = ContrastModel.from_pretrained(
    plm_name,
    num_labels=num_class,
    contrast_loss=run_args.contrast,
    graph=run_args.graph,
    layer=run_args.layer,
    data_path=data_path,
    multi_label=run_args.multi,
    lamb=run_args.lamb,
    threshold=run_args.thre,
    plm_name=plm_name,
)
model.load_state_dict(checkpoint["param"])
model.to(device)

truth_rows = []
score_rows = []
model.eval()
with torch.no_grad():
    for data, label, idx in test_loader:
        padding_mask = data != tokenizer.pad_token_id
        output = model(data, padding_mask, return_dict=True)
        truth_rows.append(label.detach().cpu().numpy())
        score_rows.append(torch.sigmoid(output["logits"]).detach().cpu().numpy())

y_true = np.concatenate(truth_rows, axis=0).astype(np.int32)
scores = np.concatenate(score_rows, axis=0)
y_pred = (scores > threshold).astype(np.int32)

result = {
    "model": payload["model"],
    "seed": int(payload["seed"]),
    "checkpoint": str(checkpoint_path),
    "threshold": threshold,
    "rows": depth_rows(y_true, y_pred, load_levels(data_path)),
}
with open(payload["output_json"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print("LABEL_DEPTH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
"""


HILL_INLINE = r"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

repo_root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path.cwd()))

import utils
from build_hierarchy_utils import parse_label_hierarchy
from data.prepare_raehmc import parse_root_label_names, promote_named_roots
from model.contrast import ContrastModel, GraphContrast, StructureContrast
from train import BertDataset, resolve_max_token


def select_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def micro_f1_np(y_true, y_pred):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


def macro_f1_np(y_true, y_pred):
    if y_true.shape[1] == 0:
        return 0.0
    scores = []
    yt_all = y_true.astype(bool)
    yp_all = y_pred.astype(bool)
    for col in range(y_true.shape[1]):
        yt = yt_all[:, col]
        yp = yp_all[:, col]
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def depth_rows(y_true, y_pred, levels):
    labels_by_level = defaultdict(list)
    for label_id, level in levels.items():
        labels_by_level[int(level)].append(int(label_id))
    rows = []
    for level in sorted(labels_by_level):
        cols = np.array(sorted(labels_by_level[level]), dtype=int)
        yt = y_true[:, cols].astype(np.int32)
        yp = y_pred[:, cols].astype(np.int32)
        rows.append({
            "level": int(level),
            "label_count": int(len(cols)),
            "positive_count": int(yt.sum()),
            "predicted_count": int(yp.sum()),
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows


def load_levels(data_path):
    meta_path = Path(data_path) / "dataset_meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(meta["hierarchy_json"], "r", encoding="utf-8") as handle:
        hierarchy_payload = json.load(handle)
    root_names = parse_root_label_names(meta.get("root_label_name", []))
    if bool(meta.get("exclude_root_label", True)):
        hierarchy_payload = promote_named_roots(hierarchy_payload, root_names)
    hd = parse_label_hierarchy(hierarchy_payload)
    return hd.levels


checkpoint_path = Path(payload["checkpoint"])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
run_args = checkpoint["args"]
device = select_device(payload["device"])
data_dir = getattr(run_args, "data_dir", "data")
dataset_name = getattr(run_args, "dataset")
cfg_dir = getattr(run_args, "cfg_dir", "config")
model_name = getattr(run_args, "model_name", "hill")
plm_name = payload["plm"] or getattr(run_args, "plm", "bert-base-uncased")
data_path = os.path.join(data_dir, dataset_name)
max_token = resolve_max_token(data_path, payload["max_token"])
threshold = float(payload["threshold"])

config = utils.Configure(config_json_file=os.path.join(cfg_dir, model_name + ".json"))
config.update(run_args.__dict__)
config.device_setting.device = device
config.plm = plm_name

tokenizer = AutoTokenizer.from_pretrained(plm_name)
label_dict = torch.load(os.path.join(data_path, "bert_value_dict.pt"), map_location="cpu", weights_only=False)
num_class = len(label_dict)
dataset = BertDataset(max_token=max_token, device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
split = torch.load(os.path.join(data_path, "split.pt"), map_location="cpu", weights_only=False)
test_loader = DataLoader(
    Subset(dataset, split["test"]),
    batch_size=int(payload["batch_size"]),
    shuffle=False,
    collate_fn=dataset.collate_fn,
)
models = {
    "hill": StructureContrast,
    "hgclr": ContrastModel,
    "gclr": GraphContrast,
}
model = models[config.model_name].from_pretrained(plm_name, num_labels=num_class, local_config=config)
model.load_state_dict(checkpoint["param"])
model.to(device)

truth_rows = []
score_rows = []
model.eval()
with torch.no_grad():
    for data, label, idx in test_loader:
        padding_mask = data != tokenizer.pad_token_id
        output = model(data, padding_mask, return_dict=True)
        truth_rows.append(label.detach().cpu().numpy())
        score_rows.append(torch.sigmoid(output["logits"]).detach().cpu().numpy())

y_true = np.concatenate(truth_rows, axis=0).astype(np.int32)
scores = np.concatenate(score_rows, axis=0)
y_pred = (scores > threshold).astype(np.int32)

result = {
    "model": payload["model"],
    "seed": int(payload["seed"]),
    "checkpoint": str(checkpoint_path),
    "threshold": threshold,
    "rows": depth_rows(y_true, y_pred, load_levels(data_path)),
}
with open(payload["output_json"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print("LABEL_DEPTH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
"""


HPT_INLINE = r"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

repo_root = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path.cwd()))

import utils
from build_hierarchy_utils import parse_label_hierarchy
from data.prepare_raehmc import parse_root_label_names, promote_named_roots
from models.prompt import Prompt


def select_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def micro_f1_np(y_true, y_pred):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int((yt & yp).sum())
    fp = int((~yt & yp).sum())
    fn = int((yt & ~yp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))


def macro_f1_np(y_true, y_pred):
    if y_true.shape[1] == 0:
        return 0.0
    scores = []
    yt_all = y_true.astype(bool)
    yp_all = y_pred.astype(bool)
    for col in range(y_true.shape[1]):
        yt = yt_all[:, col]
        yp = yp_all[:, col]
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def depth_rows(y_true, y_pred, levels):
    labels_by_level = defaultdict(list)
    for label_id, level in levels.items():
        labels_by_level[int(level)].append(int(label_id))
    rows = []
    for level in sorted(labels_by_level):
        cols = np.array(sorted(labels_by_level[level]), dtype=int)
        yt = y_true[:, cols].astype(np.int32)
        yp = y_pred[:, cols].astype(np.int32)
        rows.append({
            "level": int(level),
            "label_count": int(len(cols)),
            "positive_count": int(yt.sum()),
            "predicted_count": int(yp.sum()),
            "micro_f1": micro_f1_np(yt, yp),
            "macro_f1": macro_f1_np(yt, yp),
        })
    return rows


def load_levels(data_path):
    meta_path = Path(data_path) / "dataset_meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(meta["hierarchy_json"], "r", encoding="utf-8") as handle:
        hierarchy_payload = json.load(handle)
    root_names = parse_root_label_names(meta.get("root_label_name", []))
    if bool(meta.get("exclude_root_label", True)):
        hierarchy_payload = promote_named_roots(hierarchy_payload, root_names)
    hd = parse_label_hierarchy(hierarchy_payload)
    return hd.levels


checkpoint_path = Path(payload["checkpoint"])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
run_args = checkpoint["args"]
device = select_device(payload["device"])
utils.seed_torch(int(run_args.seed))
data_path = os.path.join("data", run_args.data)
tokenizer = AutoTokenizer.from_pretrained(run_args.arch)

label_dict = torch.load(os.path.join(data_path, "value_dict.pt"), map_location="cpu", weights_only=False)
slot2value = torch.load(os.path.join(data_path, "slot.pt"), map_location="cpu", weights_only=False)
num_class = len(label_dict)
value2slot = {}
for parent_id, child_ids in slot2value.items():
    for child_id in child_ids:
        value2slot[int(child_id)] = int(parent_id)
for label_id in range(num_class):
    value2slot.setdefault(label_id, -1)


def get_depth(label_id):
    depth = 0
    while value2slot[label_id] != -1:
        depth += 1
        label_id = value2slot[label_id]
    return depth


depth_dict = {label_id: get_depth(label_id) for label_id in range(num_class)}
max_depth = max(depth_dict.values()) + 1
depth2label = {
    depth: [label_id for label_id, label_depth in depth_dict.items() if label_depth == depth]
    for depth in range(max_depth)
}
path_list = [(parent_id, child_id) for child_id, parent_id in value2slot.items() if parent_id != -1]
for depth, label_ids in depth2label.items():
    for label_id in label_ids:
        path_list.append((num_class + depth, label_id))

dataset = datasets.load_from_disk(os.path.join(data_path, run_args.model))
dataset["test"].set_format("torch", columns=["attention_mask", "input_ids", "labels"])
test_loader = DataLoader(dataset["test"], batch_size=int(payload["batch_size"]), shuffle=False)

model = Prompt.from_pretrained(
    run_args.arch,
    num_labels=num_class,
    path_list=path_list,
    layer=run_args.layer,
    graph_type=run_args.graph,
    data_path=data_path,
    depth2label=depth2label,
)
model.init_embedding()
model.load_state_dict(checkpoint["param"])
model.to(device)
model.eval()

truth_rows = []
prediction_rows = []
with torch.no_grad():
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output_ids, _ = model.generate(batch["input_ids"], depth2label=depth2label)
        labels = batch["labels"].view(-1, max_depth, num_class)
        y_true_batch = (labels == 1).any(dim=1).detach().cpu().numpy().astype(np.int32)
        y_pred_batch = np.zeros((len(output_ids), num_class), dtype=np.int32)
        for row_index, label_ids in enumerate(output_ids):
            y_pred_batch[row_index, list(set(label_ids))] = 1
        truth_rows.append(y_true_batch)
        prediction_rows.append(y_pred_batch)

y_true = np.concatenate(truth_rows, axis=0)
y_pred = np.concatenate(prediction_rows, axis=0)
result = {
    "model": payload["model"],
    "seed": int(payload["seed"]),
    "checkpoint": str(checkpoint_path),
    "threshold": 0.0,
    "rows": depth_rows(y_true, y_pred, load_levels(data_path)),
}
with open(payload["output_json"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print("LABEL_DEPTH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)
"""


PLOT_INLINE = r"""
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

with summary_csv.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

by_model = {}
for row in rows:
    by_model.setdefault(str(row["model"]), []).append(row)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
metric_specs = [
    ("micro_f1_mean", "Micro-F1"),
    ("macro_f1_mean", "Macro-F1"),
]
all_levels = sorted({int(row["level"]) for row in rows})
for ax, (mean_key, title) in zip(axes, metric_specs):
    for model in model_order:
        model_label = model_labels.get(model, model)
        model_rows = sorted(by_model.get(model_label, []), key=lambda row: int(row["level"]))
        if not model_rows:
            continue
        levels = [int(row["level"]) for row in model_rows]
        means = [float(row[mean_key]) for row in model_rows]
        ax.plot(levels, means, marker="o", linewidth=2, label=model_label)
    ax.set_title(title)
    ax.set_xlabel("Label depth")
    ax.set_ylabel("F1 score")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if all_levels:
        ax.set_xticks(all_levels)
axes[-1].legend(loc="best")
fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=220)
plt.close(fig)
"""


MODEL_LABELS = {
    "bert_ft": "BERT-FT",
    "raehmc": "RAE-HMC",
    "hgclr": "HGCLR",
    "hill": "HILL",
    "hpt": "HPT",
}


def default_env_python(env_name: str) -> str:
    candidate = Path.home() / ".conda" / "envs" / env_name / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BERT-FT, RAE-HMC, HGCLR, HILL, and HPT checkpoints by label depth, "
            "then save per-seed CSVs and a F1-vs-depth plot."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert_ft", "raehmc", "hgclr", "hill", "hpt"],
        choices=list(MODEL_LABELS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list. Overrides start/end.")
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--seed-end", type=int, default=45)
    parser.add_argument("--dataset-base", default="raehmc_food")
    parser.add_argument("--dataset-csv", default="dataset/dataset.csv", help="Dataset CSV used for RAE-HMC evaluation.")
    parser.add_argument(
        "--hierarchy-json",
        default="dataset/label_hierarchy.json",
        help="Hierarchy JSON used for RAE-HMC evaluation.",
    )
    parser.add_argument("--temp-prefix", default="baselinetmp", help="Used by the default BERT-FT checkpoint template.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plm", default=None, help="Override checkpoint PLM for HGCLR/HILL. Defaults to checkpoint args.")
    parser.add_argument("--max-token", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for HGCLR/HILL sigmoid scores.")
    parser.add_argument("--hgclr-eval-checkpoint", default="_macro", choices=["_macro", "_micro"])
    parser.add_argument("--hill-eval-checkpoint", default="macro", choices=["macro", "micro"])
    parser.add_argument("--hpt-eval-checkpoint", default="_macro", choices=["_macro", "_micro"])
    parser.add_argument("--hpt-temp-prefix", default="hptdepth")
    parser.add_argument("--bert-ft-python", default=default_env_python("RaehmcEnv"))
    parser.add_argument("--raehmc-python", default=default_env_python("RaehmcEnv"))
    parser.add_argument("--hgclr-python", default=default_env_python("hgclr-gpu"))
    parser.add_argument("--hill-python", default=default_env_python("hill-gpu"))
    parser.add_argument("--hpt-python", default=default_env_python("hpt-gpu"))
    parser.add_argument(
        "--bert-ft-checkpoint-template",
        default="baselines/BERT-FT/checkpoints/{temp_prefix}_{dataset_base}_s{seed}/best_model.pt",
        help="Template supports {repo_root}, {dataset_base}, {temp_prefix}, and {seed}.",
    )
    parser.add_argument(
        "--raehmc-checkpoint-template",
        default="outputs/ablation_cl/ablation_cl_runs/seed{seed}/raehmc_cl/best_model_holdout.pt",
        help="Template supports {repo_root}, {dataset_base}, and {seed}.",
    )
    parser.add_argument(
        "--hgclr-checkpoint-template",
        default=(
            "baselines/HGCLR/checkpoints/"
            "{dataset_base}_s{seed}-multiseed_s{seed}/checkpoint_best{hgclr_eval_checkpoint}.pt"
        ),
        help="Template supports {repo_root}, {dataset_base}, {seed}, and {hgclr_eval_checkpoint}.",
    )
    parser.add_argument(
        "--hill-checkpoint-template",
        default="baselines/HILL/ckpt/{dataset_base}_s{seed}-multiseed_s{seed}/best_{hill_eval_checkpoint}.pt",
        help="Template supports {repo_root}, {dataset_base}, {seed}, and {hill_eval_checkpoint}.",
    )
    parser.add_argument(
        "--hpt-checkpoint-template",
        default=(
            "baselines/HPT/checkpoints/"
            "{hpt_temp_prefix}_{dataset_base}_s{seed}-{hpt_temp_prefix}_s{seed}/"
            "checkpoint_best{hpt_eval_checkpoint}.pt"
        ),
        help=(
            "Template supports {repo_root}, {dataset_base}, {seed}, "
            "{hpt_temp_prefix}, and {hpt_eval_checkpoint}."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/baselines_label_depth")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--y-min", type=float, default=0.8, help="Minimum y-axis value for the F1 plot.")
    parser.add_argument("--y-max", type=float, default=1.0, help="Maximum y-axis value for the F1 plot.")
    parser.add_argument(
        "--plot-python",
        default=default_env_python("RaehmcEnv"),
        help="Python used as a fallback for plotting when the current Python lacks matplotlib.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-model/seed JSON results.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Fail instead of skipping missing checkpoints.")
    parser.add_argument("--no-plot", action="store_true", help="Write CSVs without generating the figure.")
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return list(args.seeds)
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def ensure_python_exists(python_path: str, label: str) -> None:
    if os.path.isabs(python_path) or os.path.sep in python_path:
        if not Path(python_path).exists():
            raise FileNotFoundError(f"{label} Python not found: {python_path}")
        return
    if shutil.which(python_path) is None:
        raise FileNotFoundError(f"{label} Python not found on PATH: {python_path}")


def template_path(template: str, args: argparse.Namespace, repo_root: Path, seed: int) -> Path:
    rendered = template.format(
        repo_root=str(repo_root),
        dataset_base=args.dataset_base,
        temp_prefix=args.temp_prefix,
        seed=int(seed),
        hgclr_eval_checkpoint=args.hgclr_eval_checkpoint,
        hill_eval_checkpoint=args.hill_eval_checkpoint,
        hpt_eval_checkpoint=args.hpt_eval_checkpoint,
        hpt_temp_prefix=args.hpt_temp_prefix,
    )
    path = Path(rendered)
    return path if path.is_absolute() else repo_root / path


def checkpoint_for(model: str, args: argparse.Namespace, repo_root: Path, seed: int) -> Path:
    if model == "bert_ft":
        return template_path(args.bert_ft_checkpoint_template, args, repo_root, seed)
    if model == "raehmc":
        return template_path(args.raehmc_checkpoint_template, args, repo_root, seed)
    if model == "hgclr":
        return template_path(args.hgclr_checkpoint_template, args, repo_root, seed)
    if model == "hill":
        return template_path(args.hill_checkpoint_template, args, repo_root, seed)
    if model == "hpt":
        return template_path(args.hpt_checkpoint_template, args, repo_root, seed)
    raise ValueError(f"Unknown model: {model}")


def python_for(model: str, args: argparse.Namespace) -> str:
    if model == "bert_ft":
        return args.bert_ft_python
    if model == "raehmc":
        return args.raehmc_python
    if model == "hgclr":
        return args.hgclr_python
    if model == "hill":
        return args.hill_python
    if model == "hpt":
        return args.hpt_python
    raise ValueError(f"Unknown model: {model}")


def cwd_for(model: str, repo_root: Path) -> Path:
    if model == "bert_ft":
        return repo_root / "baselines" / "BERT-FT"
    if model == "raehmc":
        return repo_root
    if model == "hgclr":
        return repo_root / "baselines" / "HGCLR"
    if model == "hill":
        return repo_root / "baselines" / "HILL"
    if model == "hpt":
        return repo_root / "baselines" / "HPT"
    raise ValueError(f"Unknown model: {model}")


def inline_for(model: str) -> str:
    if model == "bert_ft":
        return BERT_FT_INLINE
    if model == "raehmc":
        return RAEHMC_INLINE
    if model == "hgclr":
        return HGCLR_INLINE
    if model == "hill":
        return HILL_INLINE
    if model == "hpt":
        return HPT_INLINE
    raise ValueError(f"Unknown model: {model}")


def run_command(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return proc.stdout or ""


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
        "model": MODEL_LABELS[model],
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
    }
    cmd = [
        python_for(model, args),
        "-c",
        inline_for(model),
        str(repo_root),
        json.dumps(payload, ensure_ascii=False),
    ]
    run_command(cmd, cwd_for(model, repo_root))
    with output_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_result(result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for row in result["rows"]:
        rows.append({
            "model": result["model"],
            "seed": int(result["seed"]),
            "level": int(row["level"]),
            "label_count": int(row["label_count"]),
            "positive_count": int(row["positive_count"]),
            "predicted_count": int(row["predicted_count"]),
            "micro_f1": float(row["micro_f1"]),
            "macro_f1": float(row["macro_f1"]),
            "threshold": float(result["threshold"]),
            "checkpoint": result["checkpoint"],
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), int(row["level"])), []).append(row)
    summary = []
    for (model, level), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        micro_values = [float(row["micro_f1"]) for row in group]
        macro_values = [float(row["macro_f1"]) for row in group]
        label_counts = sorted({int(row["label_count"]) for row in group})
        summary.append({
            "model": model,
            "level": level,
            "seed_count": len(group),
            "label_count": label_counts[0] if len(label_counts) == 1 else ";".join(map(str, label_counts)),
            "micro_f1_mean": mean(micro_values),
            "micro_f1_std": std(micro_values),
            "macro_f1_mean": mean(macro_values),
            "macro_f1_std": std(macro_values),
        })
    return summary


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
    for row in summary_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    metric_specs = [
        ("micro_f1_mean", "Micro-F1"),
        ("macro_f1_mean", "Macro-F1"),
    ]
    for ax, (mean_key, title) in zip(axes, metric_specs):
        for model in model_order:
            model_label = MODEL_LABELS.get(model, model)
            rows = sorted(by_model.get(model_label, []), key=lambda row: int(row["level"]))
            if not rows:
                continue
            levels = [int(row["level"]) for row in rows]
            means = [float(row[mean_key]) for row in rows]
            ax.plot(levels, means, marker="o", linewidth=2, label=model_label)
        ax.set_title(title)
        ax.set_xlabel("Label depth")
        ax.set_ylabel("F1 score")
        ax.set_ylim(float(y_min), float(y_max))
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        all_levels = sorted({int(row["level"]) for row in summary_rows})
        if all_levels:
            ax.set_xticks(all_levels)
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
    ensure_python_exists(plot_python, "plot")
    cmd = [
        plot_python,
        "-c",
        PLOT_INLINE,
        str(summary_csv),
        str(plot_path),
        json.dumps(model_order, ensure_ascii=False),
        json.dumps(MODEL_LABELS, ensure_ascii=False),
        str(float(y_min)),
        str(float(y_max)),
    ]
    run_command(cmd, Path.cwd())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    seeds = resolve_seeds(args)

    for model in args.models:
        ensure_python_exists(python_for(model, args), MODEL_LABELS[model])

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "per_seed_json"
    result_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "baselines_label_depth.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "baselines_label_depth_summary.csv"
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "baselines_label_depth.png"
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
            checkpoint_path = checkpoint_for(model, args, repo_root, seed)
            output_json = result_dir / f"{model}_seed{seed}.json"
            if not checkpoint_path.exists():
                message = f"{MODEL_LABELS[model]} seed={seed}: missing checkpoint {checkpoint_path}"
                if args.fail_on_missing:
                    raise FileNotFoundError(message)
                print(f"  - skipping {message}", flush=True)
                missing.append(message)
                continue
            if args.resume and output_json.exists():
                print(f"  - reusing {MODEL_LABELS[model]} result: {output_json}", flush=True)
                with output_json.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
            else:
                print(f"  - evaluating {MODEL_LABELS[model]}: {checkpoint_path}", flush=True)
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
        raise RuntimeError("No label-depth results were produced. Check checkpoint paths or use --fail-on-missing.")

    row_fields = [
        "model",
        "seed",
        "level",
        "label_count",
        "positive_count",
        "predicted_count",
        "micro_f1",
        "macro_f1",
        "threshold",
        "checkpoint",
    ]
    write_csv(output_csv, all_rows, row_fields)
    summary_rows = summarize_rows(all_rows)
    summary_fields = [
        "model",
        "level",
        "seed_count",
        "label_count",
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
            print(
                f"Current Python lacks matplotlib; retrying plot with {args.plot_python}",
                flush=True,
            )
            plot_summary_subprocess(summary_csv, plot_path, args.models, args.plot_python, args.y_min, args.y_max)

    print(f"\nSaved per-seed label-depth CSV: {output_csv}")
    print(f"Saved seed-averaged summary CSV: {summary_csv}")
    if not args.no_plot:
        print(f"Saved label-depth plot: {plot_path}")
    if missing:
        print("\nMissing checkpoints skipped:")
        for message in missing:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
