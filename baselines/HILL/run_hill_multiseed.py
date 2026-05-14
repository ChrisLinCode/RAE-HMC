import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


METRIC_RE = re.compile(r"micro-f1:\s*([0-9]*\.?[0-9]+)\s+macro-f1:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HILL across multiple seeds and summarize test metrics.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--dataset-base", default="raehmc_food")
    parser.add_argument("--plm", default="bert-base-chinese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lamda", type=float, default=0.05)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--hidden-dropout", type=float, default=0.5)
    parser.add_argument("--tree-pooling-type", default="sum", choices=["root", "sum", "avg", "max"])
    parser.add_argument("--max-token", type=int, default=32)
    parser.add_argument("--model-name", default="hill", choices=["hill", "hgclr", "gclr"])
    parser.add_argument("--eval-checkpoint", default="macro", choices=["macro", "micro"])
    parser.add_argument("--run-prefix", default="multiseed")
    parser.add_argument("--summary-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stop", type=int, default=None)
    return parser.parse_args()


def run_command(cmd, cwd):
    print("$", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(completed.stdout, end="")
    return completed.stdout


def parse_metrics(output_text):
    matches = METRIC_RE.findall(output_text)
    if not matches:
        raise ValueError("Could not find micro-f1/macro-f1 in command output.")
    micro, macro = matches[-1]
    return {"macro": float(macro), "micro": float(micro)}


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def main():
    args = parse_args()
    hill_root = Path(__file__).resolve().parent
    summary_dir = (hill_root / args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(args.seeds)
    for idx, seed in enumerate(args.seeds, start=1):
        dataset_name = f"{args.dataset_base}_s{seed}"
        run_name = f"{dataset_name}-{args.run_prefix}_s{seed}"

        prepare_cmd = [
            sys.executable,
            "data/prepare_raehmc.py",
            "--seed",
            str(seed),
            "--output-dir",
            str(hill_root / "data" / dataset_name),
            "--plm",
            args.plm,
            "--max-token-recommended",
            str(args.max_token),
        ]
        train_cmd = [
            sys.executable,
            "train.py",
            "-d",
            dataset_name,
            "-mn",
            args.model_name,
            "-n",
            run_name,
            "-s",
            str(seed),
            "-b",
            str(args.batch_size),
            "-lr",
            str(args.learning_rate),
            "-l",
            str(args.lamda),
            "-k",
            str(args.tree_depth),
            "-hd",
            str(args.hidden_dim),
            "-dp",
            str(args.hidden_dropout),
            "-tp",
            args.tree_pooling_type,
            "--device",
            args.device,
            "--plm",
            args.plm,
            "--max_token",
            str(args.max_token),
        ]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.early_stop is not None:
            train_cmd.extend(["--early_stop", str(args.early_stop)])

        test_cmd = [
            sys.executable,
            "test.py",
            "-n",
            run_name,
            "-e",
            args.eval_checkpoint,
            "--device",
            args.device,
            "-b",
            str(args.batch_size),
            "--plm",
            args.plm,
            "--max_token",
            str(args.max_token),
        ]

        print(f"\n[{idx}/{total}] seed={seed} preparing dataset...", flush=True)
        run_command(prepare_cmd, hill_root)
        print(f"\n[{idx}/{total}] seed={seed} training...", flush=True)
        run_command(train_cmd, hill_root)
        print(f"\n[{idx}/{total}] seed={seed} testing checkpoint={args.eval_checkpoint}...", flush=True)
        test_output = run_command(test_cmd, hill_root)

        metrics = parse_metrics(test_output)
        metrics.update({
            "seed": seed,
            "dataset": dataset_name,
            "run_name": run_name,
            "eval_checkpoint": args.eval_checkpoint,
        })
        results.append(metrics)
        print(
            f"[done {idx}/{total}] seed={seed} macro={metrics['macro']:.6f} micro={metrics['micro']:.6f}",
            flush=True,
        )

    macro_values = [item["macro"] for item in results]
    micro_values = [item["micro"] for item in results]
    summary = {
        "config": {
            "seeds": args.seeds,
            "dataset_base": args.dataset_base,
            "plm": args.plm,
            "device": args.device,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lamda": args.lamda,
            "tree_depth": args.tree_depth,
            "hidden_dim": args.hidden_dim,
            "hidden_dropout": args.hidden_dropout,
            "tree_pooling_type": args.tree_pooling_type,
            "max_token": args.max_token,
            "model_name": args.model_name,
            "eval_checkpoint": args.eval_checkpoint,
            "run_prefix": args.run_prefix,
            "epochs": args.epochs,
            "early_stop": args.early_stop,
        },
        "results": results,
        "summary": {
            "macro_mean": statistics.mean(macro_values),
            "macro_std": sample_std(macro_values),
            "micro_mean": statistics.mean(micro_values),
            "micro_std": sample_std(micro_values),
        },
    }

    summary_json = summary_dir / f"{args.run_prefix}_{args.dataset_base}_summary.json"
    summary_txt = summary_dir / f"{args.run_prefix}_{args.dataset_base}_summary.txt"

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("HILL multiseed summary\n")
        handle.write(json.dumps(summary["config"], ensure_ascii=False, indent=2))
        handle.write("\n\nPer-seed results\n")
        for item in results:
            handle.write(
                f"seed={item['seed']} macro={item['macro']:.6f} micro={item['micro']:.6f} run={item['run_name']}\n"
            )
        handle.write("\nAggregate\n")
        handle.write(f"macro_mean={summary['summary']['macro_mean']:.6f}\n")
        handle.write(f"macro_std={summary['summary']['macro_std']:.6f}\n")
        handle.write(f"micro_mean={summary['summary']['micro_mean']:.6f}\n")
        handle.write(f"micro_std={summary['summary']['micro_std']:.6f}\n")

    print("\nDone.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary TXT:  {summary_txt}")
    print(
        "macro mean/std = "
        f"{summary['summary']['macro_mean']:.6f} / {summary['summary']['macro_std']:.6f}"
    )
    print(
        "micro mean/std = "
        f"{summary['summary']['micro_mean']:.6f} / {summary['summary']['micro_std']:.6f}"
    )


if __name__ == "__main__":
    main()
