import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


MACRO_MICRO_RE = re.compile(r"macro\s+([0-9]*\.?[0-9]+)\s+micro\s+([0-9]*\.?[0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HGCLR across multiple seeds and summarize test metrics.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--dataset-base", default="raehmc_food")
    parser.add_argument("--plm", default="bert-base-chinese")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lamb", type=float, default=0.05)
    parser.add_argument("--thre", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-token", type=int, default=32)
    parser.add_argument("--run-prefix", default="multiseed")
    parser.add_argument("--summary-dir", default="outputs")
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
    matches = MACRO_MICRO_RE.findall(output_text)
    if not matches:
        raise ValueError("Could not find `macro ... micro ...` in command output.")
    macro, micro = matches[-1]
    return {"macro": float(macro), "micro": float(micro)}


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def main():
    args = parse_args()
    hgclr_root = Path(__file__).resolve().parent
    summary_dir = (hgclr_root / args.summary_dir).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(args.seeds)
    for idx, seed in enumerate(args.seeds, start=1):
        dataset_name = f"{args.dataset_base}_s{seed}"
        run_name = f"{args.run_prefix}_s{seed}"
        full_run_name = f"{dataset_name}-{run_name}"

        prepare_cmd = [
            sys.executable,
            "data/prepare_raehmc.py",
            "--seed",
            str(seed),
            "--output-dir",
            str(hgclr_root / "data" / dataset_name),
            "--plm",
            args.plm,
            "--max-token-recommended",
            str(args.max_token),
        ]
        train_cmd = [
            sys.executable,
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
            str(args.batch),
            "--lamb",
            str(args.lamb),
            "--thre",
            str(args.thre),
            "--tau",
            str(args.tau),
            "--lr",
            str(args.lr),
            "--seed",
            str(seed),
            "--max-token",
            str(args.max_token),
        ]
        test_cmd = [
            sys.executable,
            "test.py",
            "--name",
            full_run_name,
            "--plm",
            args.plm,
            "--device",
            args.device,
            "--max-token",
            str(args.max_token),
        ]

        print(f"\n[{idx}/{total}] seed={seed} preparing dataset...", flush=True)
        run_command(prepare_cmd, hgclr_root)
        print(f"\n[{idx}/{total}] seed={seed} training...", flush=True)
        run_command(train_cmd, hgclr_root)
        print(f"\n[{idx}/{total}] seed={seed} testing...", flush=True)
        test_output = run_command(test_cmd, hgclr_root)
        metrics = parse_metrics(test_output)
        metrics.update(
            {
                "seed": seed,
                "dataset": dataset_name,
                "run_name": full_run_name,
            }
        )
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
            "batch": args.batch,
            "lamb": args.lamb,
            "thre": args.thre,
            "tau": args.tau,
            "lr": args.lr,
            "max_token": args.max_token,
            "run_prefix": args.run_prefix,
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
        handle.write("HGCLR multiseed summary\n")
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
