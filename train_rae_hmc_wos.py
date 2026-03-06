"""WOS46985 training entrypoint with an internal two-stage stratified split."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import train_rae_hmc as core


@dataclass
class WOSTrainConfig(core.TrainConfig):
    # Dataset
    dataset_csv: str = "dataset/web_of_science/WOS46985.csv"
    hierarchy_json: str = "dataset/web_of_science/WOS46985_hierarchy.json"
    root_label_name: List[str] = field(default_factory=lambda: ["Root"])

    # Split
    test_ratio: float = 0.2
    val_ratio: float = 0.2
    ensure_split_label_coverage: bool = True
    data_fraction: float = 1.0 #1.0
    seed: int = 42

    # Encoder / runtime
    model_name: str = "bert-base-uncased"
    max_len: int = 512 #272
    batch_size: int = 16 #8
    cache_tokens_on_gpu: bool = False
    use_bf16_amp: bool = True
    grad_checkpointing: bool = True

    # Output
    workdir: str = "./outputs/wos46985"


def make_wos46985_config() -> WOSTrainConfig:
    return WOSTrainConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAE-HMC on WOS46985.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for the two-stage split.")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Override directory for checkpoints and outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--max-len", type=int, default=None, help="Override tokenizer max sequence length.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override encoder model name for English WOS abstracts.",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=None,
        help="Override fraction of the full WOS46985 dataset to use before splitting, e.g. 1.0 or 0.5.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: WOSTrainConfig, args: argparse.Namespace) -> WOSTrainConfig:
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.workdir is not None:
        cfg.workdir = str(args.workdir)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.max_len is not None:
        cfg.max_len = int(args.max_len)
    if args.model_name is not None:
        cfg.model_name = str(args.model_name)
    if args.data_fraction is not None:
        cfg.data_fraction = normalize_data_fraction(args.data_fraction)
    return cfg


def normalize_data_fraction(value: float) -> float:
    frac = float(value)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"data_fraction must be in (0, 1], got {value}")
    return frac


def select_subset_indices(
    Y_all: np.ndarray,
    cfg: WOSTrainConfig,
) -> np.ndarray:
    total = int(len(Y_all))
    frac = normalize_data_fraction(getattr(cfg, "data_fraction", 1.0))
    if frac >= 1.0:
        return np.arange(total, dtype=int)

    keep_count = max(1, min(total, int(np.floor(total * frac + 0.5))))
    ensure_coverage = bool(getattr(cfg, "ensure_split_label_coverage", True))
    if Y_all.ndim == 2 and keep_count < int(Y_all.shape[1]):
        ensure_coverage = False

    _, keep_idx = core.iterative_stratified_split(
        Y_all,
        test_size=keep_count / float(total),
        seed=cfg.seed + 17,
        ensure_test_label_coverage=ensure_coverage,
    )
    keep_idx = np.sort(keep_idx.astype(int))
    return keep_idx


def build_two_stage_split_indices(
    Y_all: np.ndarray,
    cfg: WOSTrainConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensure_coverage = bool(getattr(cfg, "ensure_split_label_coverage", True))
    train_pool_idx, test_idx = core.iterative_stratified_split(
        Y_all,
        cfg.test_ratio,
        cfg.seed,
        ensure_test_label_coverage=ensure_coverage,
    )
    train_rel_idx, val_rel_idx = core.iterative_stratified_split(
        Y_all[train_pool_idx],
        cfg.val_ratio,
        cfg.seed + 1,
        ensure_test_label_coverage=ensure_coverage,
    )
    train_abs_idx = train_pool_idx[train_rel_idx]
    val_abs_idx = train_pool_idx[val_rel_idx]
    return train_pool_idx, train_rel_idx, val_rel_idx, train_abs_idx, val_abs_idx, test_idx


def save_split_manifest(
    cfg: WOSTrainConfig,
    subset_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    total_samples: int,
) -> str:
    manifest = {
        "dataset_csv": cfg.dataset_csv,
        "hierarchy_json": cfg.hierarchy_json,
        "seed": int(cfg.seed),
        "test_ratio": float(cfg.test_ratio),
        "val_ratio": float(cfg.val_ratio),
        "data_fraction": float(cfg.data_fraction),
        "counts": {
            "selected": int(len(subset_idx)),
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
            "total": int(total_samples),
        },
        "indices": {
            "selected": subset_idx.tolist(),
            "train": train_idx.tolist(),
            "validation": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }
    path = os.path.join(cfg.workdir, "split_indices.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def append_wos_result_txt(cfg: WOSTrainConfig, lines: List[str]) -> str:
    path = os.path.join(cfg.workdir, "wos_result.txt")
    os.makedirs(cfg.workdir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n\n")
        f.flush()
    return path


def main(
    cfg: WOSTrainConfig,
    summary: Optional[List[Dict[str, float]]] = None,
    scenario_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    ablation_mode = core.normalize_ablation_mode(getattr(cfg, "run_ablation", "off"))
    run_ablation = ablation_mode != "off"
    if run_ablation:
        summary = summary if summary is not None else []
        result_path = os.path.join(cfg.workdir, "ablation_result.txt")
        if summary == []:
            core.init_ablation_result_txt(result_path)
        scenarios: List[Tuple[str, Dict[str, object]]] = [
            ("global_only", {"use_memory": False, "use_local_branch": False, "use_global_branch": True}),
            ("local_only", {"use_memory": False, "use_local_branch": True, "use_global_branch": False}),
            ("memory_only", {"use_memory": True, "use_local_branch": False, "use_global_branch": False}),
            ("global_local", {"use_memory": False, "use_local_branch": True, "use_global_branch": True}),
            ("all", {"use_memory": True, "use_local_branch": True, "use_global_branch": True}),
        ]

        auto_tune_params = bool(getattr(cfg, "auto_tune_params", True))
        if ablation_mode == "fixed":
            auto_tune_params = False
        elif ablation_mode == "best":
            auto_tune_params = True

        for name, overrides in scenarios:
            sub_cfg = replace(
                cfg,
                **overrides,
                run_ablation="off",
                auto_tune_params=auto_tune_params,
                workdir=os.path.join(cfg.workdir, name),
            )
            tune_label = "on" if auto_tune_params else "off"
            print(f"\n[Ablation] Running scenario: {name} with {overrides} (tune_params={tune_label})")
            res = main(sub_cfg, summary=summary, scenario_name=name)
            if res is not None:
                summary.append(res)
                core.append_ablation_result_txt(result_path, res)

        if summary:
            print("\n[Ablation Summary]")
            for item in summary:
                use_memory = bool(item.get("use_memory", True))
                use_global = bool(item.get("use_global_branch", True))
                use_local = bool(item.get("use_local_branch", True))
                classifier_on = use_global or use_local
                fusion_on = use_memory and classifier_on

                eta_print = f"{float(item.get('eta', 0.0)):.2f}" if fusion_on else "N/A"
                delta_print = f"{float(item.get('delta', 0.0)):.2f}" if (use_memory or classifier_on) else "N/A"
                rho_val = item.get("rho", None)
                rho_print = f"{float(rho_val):.2f}" if (use_memory and rho_val is not None) else "N/A"
                top_b_val = item.get("top_b", None)
                top_b_print = f"{int(top_b_val)}" if (use_memory and top_b_val is not None) else "N/A"
                print(
                    f"  - {item['scenario']}: eta={eta_print}, delta={delta_print}, "
                    f"rho={rho_print}, top_b={top_b_print}, micro-F1={item['micro']:.4f}, "
                    f"macro-F1(all)={item['macro_all']:.4f}"
                )
        return None

    os.makedirs(cfg.workdir, exist_ok=True)
    core.set_seed(cfg.seed)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    print(
        f"[AMP] bf16_amp={'on' if bool(getattr(cfg, 'use_bf16_amp', False)) and device.type == 'cuda' else 'off'} "
        f"| grad_checkpointing={'on' if bool(getattr(cfg, 'grad_checkpointing', False)) else 'off'}"
    )

    root_names = core.parse_root_label_names(getattr(cfg, "root_label_name", "Root"))
    if bool(getattr(cfg, "exclude_root_label", False)):
        with open(cfg.hierarchy_json, "r", encoding="utf-8") as f:
            hjson = json.load(f)
        hjson = core.promote_named_roots(hjson, root_names)
        hd = core.parse_label_hierarchy(hjson)
    else:
        hd = core.load_hierarchy_from_file(cfg.hierarchy_json)
    L = hd.num_labels
    print(f"[Hierarchy] num_labels={L}, level_sizes={hd.level_sizes}")
    level_lookup = {int(k): int(v) for k, v in hd.levels.items()}
    label_levels = [level_lookup.get(i, 1) for i in range(L)]
    same_level_map: Dict[int, List[int]] = {}
    for idx, lvl in enumerate(label_levels):
        same_level_map.setdefault(lvl, []).append(idx)

    df_all = pd.read_csv(cfg.dataset_csv).reset_index(drop=True)
    all_label_lists = [core.parse_label_cell(s) for s in df_all[cfg.labels_col].tolist()]
    if bool(getattr(cfg, "exclude_root_label", False)):
        all_label_lists = [core.strip_root_label(labs, root_names) for labs in all_label_lists]
    Y_all = np.array(core.build_multi_hot_Y(all_label_lists, hd.label2id, hd.ancestors, add_ancestors=True))

    subset_idx = select_subset_indices(Y_all, cfg)
    df_selected = df_all.iloc[subset_idx].reset_index(drop=True)
    Y_selected = Y_all[subset_idx]
    print(
        f"[Data] Selected {len(subset_idx)} / {len(df_all)} samples "
        f"(data_fraction={cfg.data_fraction:.4f})"
    )

    train_pool_idx, train_rel_idx, val_rel_idx, train_idx, val_idx, test_idx = build_two_stage_split_indices(Y_selected, cfg)
    split_manifest_path = save_split_manifest(
        cfg,
        subset_idx=subset_idx,
        train_idx=subset_idx[train_idx],
        val_idx=subset_idx[val_idx],
        test_idx=subset_idx[test_idx],
        total_samples=len(df_all),
    )
    print(
        f"[Data] Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)} "
        f"(two-stage stratified split, seed={cfg.seed})"
    )
    print(f"[Save] Split indices saved to {split_manifest_path}")

    df_train_pool = df_selected.iloc[train_pool_idx].reset_index(drop=True)
    df_test = df_selected.iloc[test_idx].reset_index(drop=True)
    train_pool_texts = df_train_pool[cfg.text_col].astype(str).tolist()
    test_texts = df_test[cfg.text_col].astype(str).tolist()

    Y_tr_full = torch.tensor(Y_selected[train_pool_idx], dtype=torch.float32)
    Y_te = torch.tensor(Y_selected[test_idx], dtype=torch.float32)

    print("[Stage] Initializing shared encoder...")
    enc = core.SharedEncoder(
        core.EncoderConfig(
            model_name=cfg.model_name,
            max_length=cfg.max_len,
            pooling=cfg.encoder_pooling,
            normalize=True,
            device=device_str,
        )
    )
    print("[Stage] Tokenizing train/test/label texts...")
    label_descs = core.build_label_descriptions(hd, getattr(cfg, "label_path_depth", 1))
    train_tokens = core.tokenize_texts(enc.tokenizer, train_pool_texts, cfg.max_len)
    test_tokens = core.tokenize_texts(enc.tokenizer, test_texts, cfg.max_len)
    label_tokens = core.tokenize_texts(enc.tokenizer, label_descs, cfg.max_len)
    del enc

    level_slices = core.make_level_slices(hd.levels)
    edges_pc = [(int(p), int(c)) for (p, c) in hd.edges_parent_child]
    hierarchy_obj = core.Hierarchy(num_labels=L, ancestors=hd.ancestors)

    memory_backend = "faiss_ip"
    print(f"[Memory] backend={memory_backend}")
    mem_cfg = core.MemoryConfig(
        backend=memory_backend,
        top_b=cfg.top_b,
        tau_mem=cfg.tau_mem,
        rho=cfg.rho,
        device=device_str,
    )

    fold_result = core.train_single_fold(
        fold_name="holdout",
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        same_level_map=same_level_map,
        edges_pc=edges_pc,
        label_tokens=label_tokens,
        train_tokens_full=train_tokens,
        Y_train_full=Y_tr_full,
        train_indices=train_rel_idx,
        val_indices=val_rel_idx,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg,
    )

    val_metric = core.get_val_metric_name(cfg)
    metric_val = float(fold_result.get("best_val_score", -1.0))
    micro_val = float(fold_result.get("best_val_micro", -1.0))
    macro_val = float(fold_result.get("best_val_macro", -1.0))
    rho_info = fold_result.get("last_tuned_rho", None)
    eta_info = fold_result.get("last_tuned_eta", None)
    delta_info = fold_result.get("last_tuned_delta", None)
    top_b_info = fold_result.get("last_tuned_top_b", None)
    top_b_print = f", top_b={int(top_b_info)}" if top_b_info is not None else ""
    print("\n[Holdout] Validation summary:")
    if bool(fold_result.get("use_memory", True)) and rho_info is not None:
        holdout_summary = (
            f"holdout: best val {val_metric}-F1={metric_val:.4f}, micro-F1={micro_val:.4f}, "
            f"macro-F1={macro_val:.4f}, rho={rho_info:.2f}, eta={eta_info:.2f}, "
            f"delta={delta_info:.2f}{top_b_print}, checkpoint={fold_result['best_path']}"
        )
        print(f"  - {holdout_summary}")
    else:
        holdout_summary = (
            f"holdout: best val {val_metric}-F1={metric_val:.4f}, micro-F1={micro_val:.4f}, "
            f"macro-F1={macro_val:.4f}, checkpoint={fold_result['best_path']}"
        )
        print(f"  - {holdout_summary}")

    result_txt_path = append_wos_result_txt(
        cfg,
        [
            f"[{datetime.now().isoformat(timespec='seconds')}] Holdout Validation",
            f"data_fraction={cfg.data_fraction:.4f}, seed={cfg.seed}, batch_size={cfg.batch_size}, max_len={cfg.max_len}, model_name={cfg.model_name}",
            f"split_counts: selected={len(subset_idx)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            holdout_summary,
            f"split_manifest={split_manifest_path}",
        ],
    )
    print(f"[Save] Holdout summary appended to {result_txt_path}")

    eta_final = fold_result["last_tuned_eta"] if fold_result.get("last_tuned_eta") is not None else cfg.eta
    delta_final = fold_result["last_tuned_delta"] if fold_result.get("last_tuned_delta") is not None else cfg.delta
    delta_levels_final = fold_result.get("last_tuned_delta_levels", None)
    rho_final = fold_result["last_tuned_rho"] if fold_result.get("last_tuned_rho") is not None else cfg.rho
    top_b_final = fold_result["last_tuned_top_b"] if fold_result.get("last_tuned_top_b") is not None else cfg.top_b
    mem_cfg_final = replace(mem_cfg, rho=rho_final, top_b=top_b_final)

    enc, clf, clf_cfg_full = core.train_full_model(
        cfg=cfg,
        hd=hd,
        hierarchy_obj=hierarchy_obj,
        level_slices=level_slices,
        label_levels=label_levels,
        same_level_map=same_level_map,
        edges_pc=edges_pc,
        label_tokens=label_tokens,
        train_tokens_full=train_tokens,
        Y_train_full=Y_tr_full,
        device=device,
        device_str=device_str,
        mem_cfg=mem_cfg_final,
    )

    enc.eval()
    if clf is not None:
        clf.eval()

    engine = core.InferenceEngine(
        core.InferenceConfig(eta=eta_final, delta=delta_final, topk=cfg.topk, device=device_str),
        hierarchy_obj,
    )

    mem = None
    if cfg.use_memory:
        Z_eval = core.encode_with_encoder(enc, label_tokens, cfg.batch_size, device)
        X_tr_mem = core.encode_with_encoder(enc, train_tokens, cfg.batch_size, device)
        X_mem_base, Y_mem_base = core.prepare_memory_inputs(X_tr_mem, Y_tr_full, hd)
        mem = core.SemanticMemory(mem_cfg_final)
        mem.build(X_mem_base, Z_eval, Y_mem_base)

    X_te_enc = core.encode_with_encoder(enc, test_tokens, cfg.batch_size, device)
    X_te_dev = X_te_enc.to(device)
    with torch.no_grad():
        if clf is not None:
            p_cls_te = clf(X_te_dev)["p_cls"]
        else:
            p_cls_te = torch.zeros(X_te_dev.size(0), L, device=device)
        s_mem_te = mem.batch_query(X_te_dev, top_b=top_b_final) if mem is not None else torch.zeros_like(p_cls_te)
    pred_te = core.predict_with_strategy(
        s_mem=s_mem_te,
        p_cls=p_cls_te,
        engine=engine,
        cfg=cfg,
        eta_override=eta_final,
        delta_override=delta_final,
        label_levels=label_levels,
        delta_levels_override=delta_levels_final,
    )

    y_true_te = (Y_te.cpu().numpy() > 0.5).astype(np.int32)
    y_pred_te = pred_te.cpu().numpy().astype(np.int32)

    micro = core.micro_f1(y_true_te, y_pred_te)
    macro_all = core.macro_f1(y_true_te, y_pred_te)
    print("[TEST] Per-label metrics:")
    core.per_label_report(y_true_te, y_pred_te, hd.id2label)
    print(
        f"[Final Tuning] Using eta={eta_final:.2f}, delta={delta_final:.2f}, "
        f"rho={rho_final:.2f}, top_b={top_b_final} derived from holdout validation."
    )
    if core.normalize_delta_mode(cfg) == "level" and delta_levels_final:
        levels_sorted = sorted(delta_levels_final.items(), key=lambda x: x[0])
        levels_str = ", ".join([f"L{lvl}={val:.2f}" for lvl, val in levels_sorted])
        print(f"[Final Tuning] Per-level delta: {levels_str}")
    print(f"[TEST] micro-F1={micro:.4f}")
    print(f"[TEST] macro-F1(all)={macro_all:.4f}")

    export_path = os.path.join(cfg.workdir, "test_result.xlsx")
    rows = []
    id2label = {int(k): v for k, v in hd.id2label.items()}
    texts = df_test[cfg.text_col].astype(str).tolist()
    for i in range(len(texts)):
        true_ids = np.where(y_true_te[i] > 0)[0].tolist()
        pred_ids = np.where(y_pred_te[i] > 0)[0].tolist()
        true_labels = [id2label.get(t, str(t)) for t in true_ids]
        pred_labels = [id2label.get(p, str(p)) for p in pred_ids]
        fp = [id2label.get(p, str(p)) for p in pred_ids if p not in set(true_ids)]
        fn = [id2label.get(t, str(t)) for t in true_ids if t not in set(pred_ids)]
        rows.append(
            {
                "text": texts[i],
                "label": "; ".join(true_labels),
                "pred_label": "; ".join(pred_labels),
                "false_positive": "; ".join(fp),
                "false_negative": "; ".join(fn),
            }
        )
    df_out = pd.DataFrame(rows)
    try:
        df_out.to_excel(export_path, index=False)
        print(f"[Save] Test predictions saved to {export_path}")
        export_saved_path = export_path
    except Exception as exc:
        fallback = os.path.splitext(export_path)[0] + ".csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[Warn] Failed to write xlsx: {exc}. Saved CSV instead: {fallback}")
        export_saved_path = fallback

    if mem is not None:
        mem.save(os.path.join(cfg.workdir, "memory_store"))
    full_ckpt_path = os.path.join(cfg.workdir, "best_model_full.pt")
    torch.save(
        {
            "encoder_state": enc.state_dict(),
            "classifier_state": (clf.state_dict() if clf is not None else None),
            "clf_cfg": clf_cfg_full.__dict__,
            "eta": eta_final,
            "delta": delta_final,
            "top_b": top_b_final,
            "memory_only": (clf is None) and bool(cfg.use_memory),
        },
        full_ckpt_path,
    )
    print(f"[Save] Full-train checkpoint saved to {full_ckpt_path}")
    with open(os.path.join(cfg.workdir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": hd.label2id, "id2label": {int(k): v for k, v in hd.id2label.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(cfg.workdir, "ancestors.json"), "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in hd.ancestors.items()}, f, ensure_ascii=False, indent=2)
    print(f"[Done] Artifacts saved to {cfg.workdir}")

    final_result_txt_path = append_wos_result_txt(
        cfg,
        [
            f"[{datetime.now().isoformat(timespec='seconds')}] Final Test Evaluation",
            f"data_fraction={cfg.data_fraction:.4f}, seed={cfg.seed}, batch_size={cfg.batch_size}, max_len={cfg.max_len}, model_name={cfg.model_name}",
            f"split_counts: selected={len(subset_idx)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            f"holdout_checkpoint={fold_result['best_path']}",
            f"full_checkpoint={full_ckpt_path}",
            f"split_manifest={split_manifest_path}",
            f"test_predictions={export_saved_path}",
            f"test_micro_f1={micro:.4f}",
            f"test_macro_f1_all={macro_all:.4f}",
            f"eta={eta_final:.4f}, delta={delta_final:.4f}, rho={rho_final:.4f}, top_b={top_b_final}",
        ],
    )
    print(f"[Save] Final evaluation appended to {final_result_txt_path}")

    scenario = scenario_name if scenario_name is not None else os.path.basename(cfg.workdir.rstrip(os.sep))
    return {
        "scenario": scenario,
        "eta": eta_final,
        "delta": delta_final,
        "rho": (rho_final if cfg.use_memory else None),
        "top_b": (top_b_final if cfg.use_memory else None),
        "micro": micro,
        "macro_all": macro_all,
        "use_memory": bool(cfg.use_memory),
        "use_global_branch": bool(cfg.use_global_branch),
        "use_local_branch": bool(cfg.use_local_branch),
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = make_wos46985_config()
    cfg = apply_cli_overrides(cfg, args)
    main(cfg)
