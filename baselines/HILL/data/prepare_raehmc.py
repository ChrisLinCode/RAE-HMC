import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from build_hierarchy_utils import build_multi_hot_Y, parse_label_hierarchy


def parse_label_cell(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell)
    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text.strip()] if text.strip() else []


def parse_root_label_names(root_label_name: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    if root_label_name is None:
        return []

    names: List[str] = []
    if isinstance(root_label_name, str):
        names = [item.strip() for item in root_label_name.replace(";", ",").split(",") if item.strip()]
    elif isinstance(root_label_name, (list, tuple)):
        for item in root_label_name:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                names.extend([part.strip() for part in text.replace(";", ",").split(",") if part.strip()])
    else:
        text = str(root_label_name).strip()
        if text:
            names = [text]

    deduped: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _root_payload_to_dict(payload) -> Dict[str, object]:
    if isinstance(payload, dict):
        return dict(payload)

    promoted: Dict[str, object] = {}
    if isinstance(payload, list):
        for elem in payload:
            if isinstance(elem, dict):
                for key, value in elem.items():
                    promoted[key] = value
            elif isinstance(elem, str):
                promoted[elem] = None
    elif isinstance(payload, str):
        promoted[payload] = None
    return promoted


def promote_named_roots(hjson: Dict[str, object], root_names: List[str]) -> Dict[str, object]:
    if not root_names:
        return hjson

    root_set = set(root_names)
    current = dict(hjson)
    while any(key in root_set for key in current.keys()):
        promoted: Dict[str, object] = {}
        for key, value in current.items():
            if key in root_set:
                for child_key, child_value in _root_payload_to_dict(value).items():
                    if child_key not in promoted:
                        promoted[child_key] = child_value
            elif key not in promoted:
                promoted[key] = value
        if promoted == current:
            break
        current = promoted
    return current


def strip_root_label(labels: Iterable[str], root_names: List[str]) -> List[str]:
    if not root_names:
        return list(labels)
    root_set = set(root_names)
    return [label for label in labels if label not in root_set]


def iterative_stratified_split(
    y: np.ndarray,
    test_size: float,
    seed: int,
    ensure_test_label_coverage: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = y.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    test_count = max(1, int(round(n_samples * test_size)))
    available = np.ones(n_samples, dtype=bool)
    label_totals = y.sum(axis=0).astype(float)
    desired = label_totals * (test_count / max(1, n_samples))
    assigned = np.zeros_like(desired)
    test_indices: List[int] = []

    if ensure_test_label_coverage:
        for label_idx in range(y.shape[1]):
            candidates = np.where((y[:, label_idx] > 0) & available)[0]
            if len(candidates) == 0:
                continue
            chosen = int(rng.choice(candidates))
            available[chosen] = False
            test_indices.append(chosen)
            assigned += y[chosen]
        test_count = max(test_count, len(test_indices))

    while len(test_indices) < test_count and available.any():
        need = desired - assigned
        if np.all(need <= 0):
            candidates = np.where(available)[0]
        else:
            need_mask = need.copy()
            need_mask[need_mask <= 0] = -np.inf
            label_idx = int(np.argmax(need_mask))
            candidates = np.where((y[:, label_idx] > 0) & available)[0]
            if len(candidates) == 0:
                candidates = np.where(available)[0]
        if len(candidates) == 0:
            break
        chosen = int(rng.choice(candidates))
        available[chosen] = False
        test_indices.append(chosen)
        assigned += y[chosen]

    train_indices = np.where(available)[0]
    return train_indices.astype(int), np.array(test_indices, dtype=int)


def build_split_indices(y_all: np.ndarray, test_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[int]]:
    train_pool_idx, test_idx = iterative_stratified_split(y_all, test_ratio, seed, ensure_test_label_coverage=True)
    train_rel_idx, val_rel_idx = iterative_stratified_split(
        y_all[train_pool_idx], val_ratio, seed + 1, ensure_test_label_coverage=True
    )
    train_idx = train_pool_idx[train_rel_idx]
    val_idx = train_pool_idx[val_rel_idx]
    return {
        "train": train_idx.astype(int).tolist(),
        "val": val_idx.astype(int).tolist(),
        "test": test_idx.astype(int).tolist(),
    }


def write_numberized_dataset(sequences: List[List[int]], output_prefix: Path) -> None:
    txt_path = output_prefix.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as handle:
        for seq in sequences:
            handle.write(" ".join(str(token) for token in seq))
            handle.write("\n")
    torch.save(sequences, output_prefix.with_suffix(".pt"))


def main() -> None:
    default_output = Path(__file__).resolve().parent / "raehmc_food"

    parser = argparse.ArgumentParser(description="Convert the RAEHMC food dataset into HILL format.")
    parser.add_argument("--dataset-csv", default=str(REPO_ROOT / "dataset" / "dataset.csv"))
    parser.add_argument("--hierarchy-json", default=str(REPO_ROOT / "dataset" / "label_hierarchy.json"))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--labels-col", default="labels")
    parser.add_argument("--plm", default="bert-base-chinese")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-token-recommended", type=int, default=32)
    parser.add_argument("--keep-root-label", action="store_true", help="Keep root labels instead of dropping them.")
    parser.add_argument("--root-label-name", default="Root,食材")
    args = parser.parse_args()
    exclude_root_label = not args.keep_root_label

    dataset_csv = Path(args.dataset_csv)
    hierarchy_json = Path(args.hierarchy_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with hierarchy_json.open("r", encoding="utf-8") as handle:
        hierarchy_payload = json.load(handle)

    root_names = parse_root_label_names(args.root_label_name)
    if exclude_root_label:
        hierarchy_payload = promote_named_roots(hierarchy_payload, root_names)
    hierarchy = parse_label_hierarchy(hierarchy_payload)

    df = pd.read_csv(dataset_csv).reset_index(drop=True)
    texts = df[args.text_col].astype(str).tolist()
    label_lists = [parse_label_cell(value) for value in df[args.labels_col].tolist()]
    if exclude_root_label:
        label_lists = [strip_root_label(labels, root_names) for labels in label_lists]

    missing_labels = sorted({label for labels in label_lists for label in labels if label not in hierarchy.label2id})
    if missing_labels:
        raise ValueError("Labels missing from hierarchy: {}".format(", ".join(missing_labels[:20])))

    y_matrix = np.array(build_multi_hot_Y(label_lists, hierarchy.label2id, hierarchy.ancestors, add_ancestors=True))
    split = build_split_indices(y_matrix, args.test_ratio, args.val_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    tokenized_texts = [tokenizer.encode(text, truncation=True) for text in texts]
    label_token_dict = {
        idx: tokenizer.encode(hierarchy.id2label[idx], add_special_tokens=False)
        for idx in range(hierarchy.num_labels)
    }
    slot = defaultdict(set)
    for parent_id, child_id in hierarchy.edges_parent_child:
        slot[parent_id].add(child_id)

    torch.save(label_token_dict, output_dir / "bert_value_dict.pt")
    torch.save(slot, output_dir / "slot.pt")
    torch.save(split, output_dir / "split.pt")

    write_numberized_dataset(tokenized_texts, output_dir / "tok")
    write_numberized_dataset(y_matrix.tolist(), output_dir / "Y")

    meta = {
        "dataset_csv": str(dataset_csv),
        "hierarchy_json": str(hierarchy_json),
        "num_samples": len(texts),
        "num_labels": hierarchy.num_labels,
        "level_sizes": hierarchy.level_sizes,
        "plm": args.plm,
        "max_token_recommended": int(args.max_token_recommended),
        "exclude_root_label": bool(exclude_root_label),
        "root_label_name": root_names,
        "split_sizes": {key: len(value) for key, value in split.items()},
    }
    with (output_dir / "dataset_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print("Wrote HILL dataset to {}".format(output_dir))
    print("Samples: {} | Labels: {} | Split: {}".format(len(texts), hierarchy.num_labels, meta["split_sizes"]))


if __name__ == "__main__":
    main()
