import argparse
import json
import os
import shutil
import tempfile
from dataclasses import replace
from itertools import product

from train_rae_hmc import TrainConfig, main


DEFAULT_GRID = {
    "base": {
        # Add any fixed overrides here.
        #"classifier_lr_global": None,
        #"classifier_lr_local": None,
        #"classifier_lr_fusion": None,

    },
    "params": {
        "seed": [41, 42, 43, 44, 45],
        "rho_candidates": [[0.1], [0.3], [0.5], [0.7], [0.9]]
    },
    "workdir_root": None,
    "use_temp_workdir": True,
    "cleanup_workdir": True,
}


def _ensure_list(value):
    return value if isinstance(value, list) else [value]


def _format_value(value):
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            text = f"{value:.0e}"
        else:
            text = f"{value:.6g}"
    else:
        text = str(value)
    return (
        text.replace(" ", "")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
    )


def _build_tag(params):
    parts = [f"{k}={_format_value(v)}" for k, v in params.items()]
    return "__".join(parts) if parts else "default"


def _expand_grid(params):
    keys = list(params.keys())
    values = [_ensure_list(params[k]) for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))

def _row_params(row):
    return {k.split("param.", 1)[1]: v for k, v in row.items() if k.startswith("param.")}

def _format_param_items(params):
    items = []
    for k, v in params.items():
        items.append(f"{k}={_format_value(v)}")
    return ", ".join(items)


def _append_row_txt(path, params, micro, macro, dynamic):
    micro_str = "None" if micro is None else f"{micro:.4f}"
    macro_str = "None" if macro is None else f"{macro:.4f}"
    dyn_parts = []
    for k in ("eta", "delta", "rho", "top_b"):
        v = dynamic.get(k, None)
        if v is None:
            dyn_parts.append(f"{k}=None")
        else:
            dyn_parts.append(f"{k}={_format_value(v)}")
    line = (
        f"params: {_format_param_items(params)} | "
        f"micro={micro_str} | macro={macro_str} | "
        + " | ".join(dyn_parts)
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _format_metric(value):
    return "None" if value is None else f"{float(value):.4f}"


def run_grid(grid_cfg, dry_run=False):
    base = dict(grid_cfg.get("base", {}))
    params = dict(grid_cfg.get("params", {}))
    workdir_root = grid_cfg.get("workdir_root")
    use_temp_workdir = bool(grid_cfg.get("use_temp_workdir", False))
    cleanup_workdir = bool(grid_cfg.get("cleanup_workdir", False))

    temp_root = None
    if use_temp_workdir:
        temp_root = tempfile.mkdtemp(prefix="raehmc_grid_")
        workdir_root = temp_root
        print(f"[Grid] Using temp workdir_root: {workdir_root}")

    combos = list(_expand_grid(params))
    if dry_run:
        print(f"[Grid] {len(combos)} combinations")
        for combo in combos:
            print(f"  - {_build_tag(combo)}")
        if temp_root and cleanup_workdir:
            shutil.rmtree(temp_root, ignore_errors=True)
        return []

    summary = []
    txt_path = grid_cfg.get("txt_path") or grid_cfg.get("csv_path") or os.path.join("outputs", "grid_search", "grid_search.txt")
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    header = "params | micro | macro | eta | delta | rho | top_b"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
    for idx, combo in enumerate(combos, start=1):
        overrides = dict(base)
        overrides.update(combo)

        if "workdir" not in overrides:
            if workdir_root:
                tag = _build_tag(combo)
                overrides["workdir"] = os.path.join(workdir_root, tag)

        cfg = replace(TrainConfig(), **overrides)
        print(f"\n[Grid] ({idx}/{len(combos)}) {combo}")
        try:
            res = main(cfg)
            status = "ok"
        except Exception as exc:
            res = None
            status = f"error: {exc}"

        row = {f"param.{k}": v for k, v in combo.items()}
        row["workdir"] = overrides.get("workdir")
        row["status"] = status
        if isinstance(res, dict):
            for k, v in res.items():
                row[f"result.{k}"] = v
        summary.append(row)
        micro_val = res.get("micro") if isinstance(res, dict) else None
        macro_val = res.get("macro_all") if isinstance(res, dict) else None
        dynamic = res if isinstance(res, dict) else {}
        _append_row_txt(txt_path, combo, micro_val, macro_val, dynamic)

        if isinstance(res, dict):
            micro = res.get("micro")
            macro = res.get("macro_all")
            print(
                f"[Grid Result] {combo} micro={_format_metric(micro)} "
                f"macro_all={_format_metric(macro)} workdir={row['workdir']}"
            )
        else:
            print(f"[Grid Result] {combo} status={status}")

    if summary:
        print("\n[Grid Summary]")
        for row in summary:
            params = _row_params(row)
            micro = row.get("result.micro")
            macro = row.get("result.macro_all")
            status = row.get("status")
            print(
                f"  - {params} micro={_format_metric(micro)} "
                f"macro_all={_format_metric(macro)} status={status}"
            )

        valid = [r for r in summary if r.get("result.macro_all") is not None]
        if valid:
            best_macro = max(valid, key=lambda r: r.get("result.macro_all", float("-inf")))
            best_micro = max(valid, key=lambda r: r.get("result.micro", float("-inf")))
            print("\n[Grid Best]")
            print(
                f"  - best_macro: {_row_params(best_macro)} "
                f"macro_all={_format_metric(best_macro.get('result.macro_all'))}"
            )
            print(
                f"  - best_micro: {_row_params(best_micro)} "
                f"micro={_format_metric(best_micro.get('result.micro'))}"
            )
    if temp_root and cleanup_workdir:
        print(f"\n[Grid] Cleaning up temp workdir_root: {temp_root}")
        shutil.rmtree(temp_root, ignore_errors=True)
    return summary


def main_cli():
    parser = argparse.ArgumentParser(description="Grid search for TrainConfig.")
    parser.add_argument("--grid", help="Path to a JSON grid config.")
    parser.add_argument("--dry-run", action="store_true", help="List combinations only.")
    args = parser.parse_args()

    if args.grid:
        with open(args.grid, "r", encoding="utf-8") as f:
            grid_cfg = json.load(f)
    else:
        grid_cfg = DEFAULT_GRID

    run_grid(grid_cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main_cli()
