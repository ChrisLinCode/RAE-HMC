import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


RAEHMC_RESULT_PREFIX = "BASELINE_RESULT_JSON="
REMOVED_COLUMNS = {"raehmc_no_cl_micro", "raehmc_no_cl_macro"}
BERT_FT_RESULT_PREFIX = "BERT_FT_RESULT_JSON="
HGCLR_METRIC_RE = re.compile(r"macro\s+([0-9]*\.?[0-9]+)\s+micro\s+([0-9]*\.?[0-9]+)")
HILL_METRIC_RE = re.compile(r"micro-f1:\s*([0-9]*\.?[0-9]+)\s+macro-f1:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)

RAEHMC_INLINE = r"""
import json
import os
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
temp_root = Path(sys.argv[2]).resolve()
payload = json.loads(sys.argv[3])

sys.path.insert(0, str(repo_root))
os.chdir(temp_root)

from train_rae_hmc import TrainConfig, main

cfg = TrainConfig()
for key, value in payload.items():
    setattr(cfg, key, value)

result = main(cfg)
if result is None:
    raise RuntimeError("train_rae_hmc.main returned None.")

print(
    "BASELINE_RESULT_JSON=" + json.dumps(
        {
            "micro": float(result["micro"]),
            "macro": float(result["macro_all"]),
        },
        ensure_ascii=False,
    )
)
"""


def default_env_python(env_name: str) -> str:
    candidate = Path.home() / ".conda" / "envs" / env_name / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed baseline evaluations for BERT-FT, RAE-HMC, HGCLR, and HILL, then save a CSV for paired tests."
    )
    parser.add_argument("--run-bert-ft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-raehmc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RAE-HMC with the current TrainConfig defaults from train_rae_hmc.py.",
    )
    parser.add_argument("--run-hgclr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-hill", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit seed list. Overrides seed-start/end.")
    parser.add_argument("--seed-start", type=int, default=41)
    parser.add_argument("--seed-end", type=int, default=70)

    parser.add_argument("--dataset-base", default="raehmc_food")
    parser.add_argument("--plm", default="bert-base-chinese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-token",
        type=int,
        default=32,
        help="Max token length for HGCLR/HILL preprocessing. RAE-HMC uses TrainConfig.max_len.",
    )
    parser.add_argument("--temp-prefix", default="baselinetmp")
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--append-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse rows already present in the output CSV and only run missing seed/model cells.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rerun enabled model cells even if the output CSV already has values, then overwrite those values.",
    )

    parser.add_argument("--raehmc-python", default=default_env_python("RaehmcEnv"))
    parser.add_argument("--bert-ft-python", default=default_env_python("RaehmcEnv"))
    parser.add_argument("--bert-ft-batch-size", type=int, default=None)
    parser.add_argument("--bert-ft-epochs", type=int, default=None)
    parser.add_argument("--bert-ft-patience", type=int, default=None)
    parser.add_argument("--hgclr-python", default=default_env_python("hgclr-gpu"))
    parser.add_argument("--hgclr-batch", type=int, default=16)
    parser.add_argument("--hgclr-lamb", type=float, default=0.05)
    parser.add_argument("--hgclr-thre", type=float, default=0.02)
    parser.add_argument("--hgclr-tau", type=float, default=1.0)
    parser.add_argument("--hgclr-lr", type=float, default=3e-5)
    parser.add_argument("--hgclr-eval-checkpoint", default="_macro", choices=["_macro", "_micro"])

    parser.add_argument("--hill-python", default=default_env_python("hill-gpu"))
    parser.add_argument("--hill-batch-size", type=int, default=16)
    parser.add_argument("--hill-learning-rate", type=float, default=1e-3)
    parser.add_argument("--hill-lamda", type=float, default=0.05)
    parser.add_argument("--hill-tree-depth", type=int, default=3)
    parser.add_argument("--hill-hidden-dim", type=int, default=768)
    parser.add_argument("--hill-hidden-dropout", type=float, default=0.5)
    parser.add_argument("--hill-tree-pooling-type", default="sum", choices=["root", "sum", "avg", "max"])
    parser.add_argument("--hill-model-name", default="hill", choices=["hill", "hgclr", "gclr"])
    parser.add_argument("--hill-eval-checkpoint", default="macro", choices=["macro", "micro"])
    parser.add_argument("--hill-epochs", type=int, default=None)
    parser.add_argument("--hill-early-stop", type=int, default=None)

    parser.add_argument(
        "--output-csv",
        default=None,
        help="Default: outputs/baselines_multiseed/baselines_multiseed.csv",
    )
    args = parser.parse_args()
    return args


def resolve_seeds(args) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")
    return list(range(args.seed_start, args.seed_end + 1))


def ensure_python_exists(python_bin: str, label: str):
    python_path = Path(python_bin)
    if python_path.is_absolute():
        if not python_path.exists():
            raise FileNotFoundError(f"{label} python not found: {python_bin}")
        return
    if shutil.which(python_bin) is None:
        raise FileNotFoundError(f"{label} python not found on PATH: {python_bin}")


def run_command(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-80:])
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"exit_code={completed.returncode}\n"
            f"last_output=\n{tail}"
        )
    return output


def write_csv(csv_path: Path, fieldnames: list[str], rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for fieldname in fieldnames:
                if fieldname.endswith(("_micro", "_macro")) and row.get(fieldname, "") not in ("", None):
                    row[fieldname] = f"{float(row[fieldname]):.4f}"
            writer.writerow(row)


def load_existing_csv(csv_path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    if not csv_path.exists():
        return [], {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in REMOVED_COLUMNS]
        rows: dict[int, dict[str, str]] = {}
        for raw_row in reader:
            if not raw_row:
                continue
            if "seed" not in raw_row or raw_row["seed"] in ("", None):
                continue
            seed = int(raw_row["seed"])
            rows[seed] = {
                key: (value if value is not None else "")
                for key, value in raw_row.items()
                if key not in REMOVED_COLUMNS
            }
            rows[seed]["seed"] = str(seed)
    return fieldnames, rows


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def parse_raehmc_metrics(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith(RAEHMC_RESULT_PREFIX):
            payload = json.loads(line[len(RAEHMC_RESULT_PREFIX):])
            return {"micro": float(payload["micro"]), "macro": float(payload["macro"])}
    raise ValueError("Could not find RAE-HMC result JSON in subprocess output.")


def parse_bert_ft_metrics(output: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        if line.startswith(BERT_FT_RESULT_PREFIX):
            payload = json.loads(line[len(BERT_FT_RESULT_PREFIX):])
            return {"micro": float(payload["micro"]), "macro": float(payload["macro"])}
    match = HILL_METRIC_RE.findall(output)
    if match:
        micro, macro = match[-1]
        return {"micro": float(micro), "macro": float(macro)}
    raise ValueError("Could not parse BERT-FT metrics from output.")


def parse_hgclr_metrics(output: str) -> dict[str, float]:
    matches = HGCLR_METRIC_RE.findall(output)
    if not matches:
        raise ValueError("Could not parse HGCLR metrics from output.")
    macro, micro = matches[-1]
    return {"micro": float(micro), "macro": float(macro)}


def parse_hill_metrics(output: str) -> dict[str, float]:
    matches = HILL_METRIC_RE.findall(output)
    if not matches:
        raise ValueError("Could not parse HILL metrics from output.")
    micro, macro = matches[-1]
    return {"micro": float(micro), "macro": float(macro)}


def run_raehmc(seed: int, args, repo_root: Path) -> dict[str, float]:
    temp_parent = repo_root / "outputs" / "baselines_multiseed" / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"raehmc_s{seed}_", dir=str(temp_parent)))
    temp_workdir = temp_root / "raehmc_artifacts"
    temp_workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_csv": str((repo_root / "dataset" / "dataset.csv").resolve()),
        "hierarchy_json": str((repo_root / "dataset" / "label_hierarchy.json").resolve()),
        "workdir": str(temp_workdir.resolve()),
        "seed": int(seed),
    }
    cmd = [
        args.raehmc_python,
        "-c",
        RAEHMC_INLINE,
        str(repo_root),
        str(temp_root),
        json.dumps(payload, ensure_ascii=False),
    ]
    try:
        output = run_command(cmd, repo_root)
        return parse_raehmc_metrics(output)
    finally:
        if args.cleanup:
            shutil.rmtree(temp_root, ignore_errors=True)


def run_bert_ft(seed: int, args, repo_root: Path) -> dict[str, float]:
    bert_root = repo_root / "baselines" / "BERT-FT"
    run_name = f"{args.temp_prefix}_{args.dataset_base}_s{seed}"
    checkpoint_dir = bert_root / "checkpoints" / run_name

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    train_cmd = [
        args.bert_ft_python,
        "train.py",
        "--dataset-csv",
        str((repo_root / "dataset" / "dataset.csv").resolve()),
        "--hierarchy-json",
        str((repo_root / "dataset" / "label_hierarchy.json").resolve()),
        "--output-dir",
        str(checkpoint_dir),
        "--seed",
        str(seed),
        "--model-name",
        args.plm,
        "--device",
        args.device,
    ]
    if args.bert_ft_batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.bert_ft_batch_size)])
    if args.bert_ft_epochs is not None:
        train_cmd.extend(["--epochs", str(args.bert_ft_epochs)])
    if args.bert_ft_patience is not None:
        train_cmd.extend(["--patience", str(args.bert_ft_patience)])

    test_cmd = [
        args.bert_ft_python,
        "test.py",
        "--checkpoint",
        str(checkpoint_dir / "best_model.pt"),
        "--device",
        args.device,
    ]

    try:
        run_command(train_cmd, bert_root)
        output = run_command(test_cmd, bert_root)
        return parse_bert_ft_metrics(output)
    finally:
        if args.cleanup:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_hgclr(seed: int, args, repo_root: Path) -> dict[str, float]:
    hgclr_root = repo_root / "baselines" / "HGCLR"
    dataset_name = f"{args.temp_prefix}_{args.dataset_base}_s{seed}"
    run_name = f"{args.temp_prefix}_s{seed}"
    full_run_name = f"{dataset_name}-{run_name}"
    dataset_dir = hgclr_root / "data" / dataset_name
    checkpoint_dir = hgclr_root / "checkpoints" / full_run_name

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    prepare_cmd = [
        args.hgclr_python,
        "data/prepare_raehmc.py",
        "--seed",
        str(seed),
        "--output-dir",
        str(dataset_dir),
        "--plm",
        args.plm,
        "--max-token-recommended",
        str(args.max_token),
    ]
    train_cmd = [
        args.hgclr_python,
        "train.py",
        "--data",
        dataset_name,
        "--plm",
        args.plm,
        "--name",
        run_name,
        "--device",
        args.device,
        "--batch",
        str(args.hgclr_batch),
        "--lamb",
        str(args.hgclr_lamb),
        "--thre",
        str(args.hgclr_thre),
        "--tau",
        str(args.hgclr_tau),
        "--lr",
        str(args.hgclr_lr),
        "--seed",
        str(seed),
        "--max-token",
        str(args.max_token),
    ]
    test_cmd = [
        args.hgclr_python,
        "test.py",
        "--name",
        full_run_name,
        "--extra",
        args.hgclr_eval_checkpoint,
        "--plm",
        args.plm,
        "--device",
        args.device,
        "--max-token",
        str(args.max_token),
    ]

    try:
        run_command(prepare_cmd, hgclr_root)
        run_command(train_cmd, hgclr_root)
        output = run_command(test_cmd, hgclr_root)
        return parse_hgclr_metrics(output)
    finally:
        if args.cleanup:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_hill(seed: int, args, repo_root: Path) -> dict[str, float]:
    hill_root = repo_root / "baselines" / "HILL"
    dataset_name = f"{args.temp_prefix}_{args.dataset_base}_s{seed}"
    run_name = f"{dataset_name}-{args.temp_prefix}_s{seed}"
    dataset_dir = hill_root / "data" / dataset_name
    checkpoint_dir = hill_root / "ckpt" / run_name

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    prepare_cmd = [
        args.hill_python,
        "data/prepare_raehmc.py",
        "--seed",
        str(seed),
        "--output-dir",
        str(dataset_dir),
        "--plm",
        args.plm,
        "--max-token-recommended",
        str(args.max_token),
    ]
    train_cmd = [
        args.hill_python,
        "train.py",
        "-d",
        dataset_name,
        "-mn",
        args.hill_model_name,
        "-n",
        run_name,
        "-s",
        str(seed),
        "-b",
        str(args.hill_batch_size),
        "-lr",
        str(args.hill_learning_rate),
        "-l",
        str(args.hill_lamda),
        "-k",
        str(args.hill_tree_depth),
        "-hd",
        str(args.hill_hidden_dim),
        "-dp",
        str(args.hill_hidden_dropout),
        "-tp",
        args.hill_tree_pooling_type,
        "--device",
        args.device,
        "--plm",
        args.plm,
        "--max_token",
        str(args.max_token),
    ]
    if args.hill_epochs is not None:
        train_cmd.extend(["--epochs", str(args.hill_epochs)])
    if args.hill_early_stop is not None:
        train_cmd.extend(["--early_stop", str(args.hill_early_stop)])

    test_cmd = [
        args.hill_python,
        "test.py",
        "-n",
        run_name,
        "-e",
        args.hill_eval_checkpoint,
        "--device",
        args.device,
        "-b",
        str(args.hill_batch_size),
        "--plm",
        args.plm,
        "--max_token",
        str(args.max_token),
    ]

    try:
        run_command(prepare_cmd, hill_root)
        run_command(train_cmd, hill_root)
        output = run_command(test_cmd, hill_root)
        return parse_hill_metrics(output)
    finally:
        if args.cleanup:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def summarize(csv_path: Path, rows: list[dict], fieldnames: list[str]):
    print(f"\nSaved paired-test CSV: {csv_path}")
    for model_name in ("bert_ft", "raehmc", "hgclr", "hill"):
        micro_key = f"{model_name}_micro"
        macro_key = f"{model_name}_macro"
        if micro_key not in fieldnames:
            continue
        micro_values = [float(row[micro_key]) for row in rows if row.get(micro_key) not in ("", None)]
        macro_values = [float(row[macro_key]) for row in rows if row.get(macro_key) not in ("", None)]
        if not micro_values or not macro_values:
            continue
        print(
            f"{model_name}: "
            f"micro mean/std={statistics.mean(micro_values):.4f}/{sample_std(micro_values):.4f}, "
            f"macro mean/std={statistics.mean(macro_values):.4f}/{sample_std(macro_values):.4f}"
        )


def has_metrics(row: dict[str, str], model_name: str) -> bool:
    micro_key = f"{model_name}_micro"
    macro_key = f"{model_name}_macro"
    return row.get(micro_key, "") not in ("", None) and row.get(macro_key, "") not in ("", None)


def should_skip(row: dict[str, str], model_name: str, overwrite_existing: bool) -> bool:
    return has_metrics(row, model_name) and not overwrite_existing


def requested_model_columns(args) -> list[str]:
    columns = []
    if args.run_bert_ft:
        columns.extend(["bert_ft_micro", "bert_ft_macro"])
    if args.run_raehmc:
        columns.extend(["raehmc_micro", "raehmc_macro"])
    if args.run_hgclr:
        columns.extend(["hgclr_micro", "hgclr_macro"])
    if args.run_hill:
        columns.extend(["hill_micro", "hill_macro"])
    return columns


def main():
    args = parse_args()
    if not any((args.run_bert_ft, args.run_raehmc, args.run_hgclr, args.run_hill)):
        raise ValueError("At least one model must be enabled.")

    repo_root = Path(__file__).resolve().parent
    seeds = resolve_seeds(args)

    if args.run_bert_ft:
        ensure_python_exists(args.bert_ft_python, "BERT-FT")
    if args.run_raehmc:
        ensure_python_exists(args.raehmc_python, "RAE-HMC")
    if args.run_hgclr:
        ensure_python_exists(args.hgclr_python, "HGCLR")
    if args.run_hill:
        ensure_python_exists(args.hill_python, "HILL")

    if args.output_csv is None:
        output_csv = repo_root / "outputs" / "baselines_multiseed" / "baselines_multiseed.csv"
    else:
        output_csv = (repo_root / args.output_csv).resolve() if not Path(args.output_csv).is_absolute() else Path(args.output_csv)

    fieldnames = ["seed"] + requested_model_columns(args)
    existing_fieldnames: list[str] = []
    existing_rows: dict[int, dict[str, str]] = {}
    if args.append_existing:
        existing_fieldnames, existing_rows = load_existing_csv(output_csv)
        if existing_rows:
            print(f"Loaded existing CSV: {output_csv} ({len(existing_rows)} rows)")
        for name in existing_fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)
        for name in fieldnames:
            for row in existing_rows.values():
                row.setdefault(name, "")

    total = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        row = existing_rows.get(seed, {"seed": str(seed)})
        row["seed"] = str(seed)
        for name in fieldnames:
            row.setdefault(name, "")
        print(f"\n[{index}/{total}] seed={seed}")

        if args.run_bert_ft:
            if should_skip(row, "bert_ft", args.overwrite_existing):
                print("  - skipping BERT-FT (already in CSV)", flush=True)
            else:
                if has_metrics(row, "bert_ft"):
                    print("  - re-running BERT-FT (overwriting CSV)", flush=True)
                else:
                    print("  - running BERT-FT", flush=True)
                metrics = run_bert_ft(seed, args, repo_root)
                row["bert_ft_micro"] = f"{metrics['micro']:.4f}"
                row["bert_ft_macro"] = f"{metrics['macro']:.4f}"
                print(
                    f"    done: micro={metrics['micro']:.4f}, macro={metrics['macro']:.4f}",
                    flush=True,
                )

        if args.run_raehmc:
            if should_skip(row, "raehmc", args.overwrite_existing):
                print("  - skipping RAE-HMC (already in CSV)", flush=True)
            else:
                if has_metrics(row, "raehmc"):
                    print("  - re-running RAE-HMC (overwriting CSV)", flush=True)
                else:
                    print("  - running RAE-HMC", flush=True)
                metrics = run_raehmc(seed, args, repo_root)
                row["raehmc_micro"] = f"{metrics['micro']:.4f}"
                row["raehmc_macro"] = f"{metrics['macro']:.4f}"
                print(
                    f"    done: micro={metrics['micro']:.4f}, macro={metrics['macro']:.4f}",
                    flush=True,
                )

        if args.run_hgclr:
            if should_skip(row, "hgclr", args.overwrite_existing):
                print("  - skipping HGCLR (already in CSV)", flush=True)
            else:
                if has_metrics(row, "hgclr"):
                    print("  - re-running HGCLR (overwriting CSV)", flush=True)
                else:
                    print("  - running HGCLR", flush=True)
                metrics = run_hgclr(seed, args, repo_root)
                row["hgclr_micro"] = f"{metrics['micro']:.4f}"
                row["hgclr_macro"] = f"{metrics['macro']:.4f}"
                print(
                    f"    done: micro={metrics['micro']:.4f}, macro={metrics['macro']:.4f}",
                    flush=True,
                )

        if args.run_hill:
            if should_skip(row, "hill", args.overwrite_existing):
                print("  - skipping HILL (already in CSV)", flush=True)
            else:
                if has_metrics(row, "hill"):
                    print("  - re-running HILL (overwriting CSV)", flush=True)
                else:
                    print("  - running HILL", flush=True)
                metrics = run_hill(seed, args, repo_root)
                row["hill_micro"] = f"{metrics['micro']:.4f}"
                row["hill_macro"] = f"{metrics['macro']:.4f}"
                print(
                    f"    done: micro={metrics['micro']:.4f}, macro={metrics['macro']:.4f}",
                    flush=True,
                )

        existing_rows[seed] = row
        rows = [existing_rows[key] for key in sorted(existing_rows)]
        write_csv(output_csv, fieldnames, rows)

    rows = [existing_rows[key] for key in sorted(existing_rows)]
    summarize(output_csv, rows, fieldnames)


if __name__ == "__main__":
    main()
