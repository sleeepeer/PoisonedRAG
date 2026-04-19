"""
Optimized ASR evaluation runner.

Runs one `main.py` process per (dataset, attack_method) pair, with repeats handled inside
`main.py` to avoid expensive process/model/dataset reloading for each repeat.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np


ATTACK_CONFIGS = [
    ("no_attack", None),
    ("poisoned_rag", "LM_targeted"),
    ("corpus_poisoning", "corpus_poisoning"),
    ("prompt_injection", "prompt_injection"),
]


def str2bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASR evaluation quickly and reliably.")
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa", "msmarco"])
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="palm2")
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--adv_per_query", type=int, default=5)
    parser.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--query_results_dir", type=str, default="asr_eval")
    parser.add_argument("--output_dir", type=str, default="results/asr_evaluation")
    parser.add_argument("--random_targets", type=str, default="True")
    parser.add_argument("--reuse_targets_per_repeat", type=str, default="True")
    parser.add_argument("--resume", type=str, default="True")
    parser.add_argument("--timeout_sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument(
        "--live_output",
        type=str,
        default="True",
        help="Stream main.py output live to terminal while also saving logs",
    )
    parser.add_argument(
        "--hf_cache_root",
        type=str,
        default=None,
        help="Optional Hugging Face/Torch cache root. Example: E:\\hf_cache",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token (overrides env vars for child processes)",
    )
    return parser.parse_args()


def get_windows_filesystem(path):
    if os.name != "nt":
        return None
    try:
        root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
        fs_name = ctypes.create_unicode_buffer(255)
        result = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            None,
            0,
            None,
            None,
            None,
            fs_name,
            len(fs_name),
        )
        if result == 0:
            return None
        return fs_name.value
    except Exception:
        return None


def choose_ntfs_cache_root():
    candidates = [r"D:\hf_cache", r"E:\hf_cache", r"C:\hf_cache"]
    for candidate in candidates:
        drive_root = os.path.splitdrive(candidate)[0]
        if not drive_root:
            continue
        if not os.path.exists(drive_root + "\\"):
            continue
        fs_type = get_windows_filesystem(candidate)
        if fs_type is None or fs_type.upper() != "FAT32":
            return candidate
    return candidates[0]


def build_hf_env(args):
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    env.setdefault("HF_HUB_MAX_WORKERS", "1")
    cache_root = args.hf_cache_root
    if cache_root is None and os.name == "nt":
        cache_root = choose_ntfs_cache_root()

    if cache_root is not None and os.name == "nt":
        fs_type = get_windows_filesystem(cache_root)
        if fs_type is not None and fs_type.upper() == "FAT32":
            fallback = choose_ntfs_cache_root()
            if os.path.normcase(fallback) != os.path.normcase(cache_root):
                print(
                    f"Warning: requested hf_cache_root={cache_root} is FAT32 (4GB file limit). "
                    f"Using {fallback} instead."
                )
                cache_root = fallback

    if cache_root:
        hub_dir = os.path.join(cache_root, "hub")
        transformers_dir = os.path.join(cache_root, "transformers")
        torch_dir = os.path.join(cache_root, "torch")
        tmp_dir = os.path.join(cache_root, "tmp")
        os.makedirs(hub_dir, exist_ok=True)
        os.makedirs(transformers_dir, exist_ok=True)
        os.makedirs(torch_dir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)

        env["HF_HOME"] = cache_root
        env["HF_HUB_CACHE"] = hub_dir
        env["HUGGINGFACE_HUB_CACHE"] = hub_dir
        env["TRANSFORMERS_CACHE"] = transformers_dir
        env["TORCH_HOME"] = torch_dir
        env["XDG_CACHE_HOME"] = cache_root
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["TMPDIR"] = tmp_dir
        env["TEMP"] = tmp_dir
        env["TMP"] = tmp_dir

    hf_token = args.hf_token or env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    return env


def run_single_experiment(args, dataset, method_name, attack_method, run_summary_path):
    run_name = f"asr_{dataset}_{method_name}"

    cmd = [
        args.python_exec,
        "main.py",
        "--eval_model_code", args.eval_model_code,
        "--eval_dataset", dataset,
        "--split", "test",
        "--query_results_dir", args.query_results_dir,
        "--model_name", args.model_name,
        "--use_truth", "False",
        "--top_k", str(args.top_k),
        "--gpu_id", str(args.gpu_id),
        "--attack_method", attack_method if attack_method is not None else "None",
        "--adv_per_query", str(args.adv_per_query),
        "--score_function", args.score_function,
        "--repeat_times", str(args.num_repeats),
        "--M", str(args.num_questions),
        "--seed", str(args.seed),
        "--random_targets", str(args.random_targets),
        "--reuse_targets_per_repeat", str(args.reuse_targets_per_repeat),
        "--save_every_iter", "False",
        "--summary_output", run_summary_path,
        "--name", run_name,
    ]

    if args.model_config_path:
        cmd.extend(["--model_config_path", args.model_config_path])

    start = time.time()
    timeout = args.timeout_sec if args.timeout_sec > 0 else None
    live_output = str2bool(args.live_output)
    child_env = build_hf_env(args)
    print(
        "HF env:",
        f"cache={child_env.get('HF_HUB_CACHE', '<default>')}",
        f"temp={child_env.get('TMP', '<default>')}",
        f"token={'set' if (child_env.get('HF_TOKEN') or child_env.get('HUGGINGFACE_HUB_TOKEN')) else 'missing'}",
    )

    if not live_output:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=child_env)
        elapsed = time.time() - start
        return completed, elapsed, run_name

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )

    combined_lines = []
    timed_out = False

    try:
        while True:
            if proc.stdout is not None:
                line = proc.stdout.readline()
            else:
                line = ""

            if line:
                print(line, end="")
                combined_lines.append(line)

            if proc.poll() is not None:
                break

            if timeout is not None and (time.time() - start) > timeout:
                timed_out = True
                proc.kill()
                break
    finally:
        if proc.stdout is not None:
            tail = proc.stdout.read()
            if tail:
                print(tail, end="")
                combined_lines.append(tail)

    elapsed = time.time() - start

    if timed_out:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    completed = SimpleNamespace(
        returncode=proc.returncode,
        stdout="".join(combined_lines),
        stderr="",
    )

    return completed, elapsed, run_name


def summary_matches_args(summary, args, dataset, attack_method):
    expected_attack = attack_method if attack_method is not None else "None"
    checks = [
        summary.get("dataset") == dataset,
        summary.get("attack_method") == expected_attack,
        int(summary.get("M", -1)) == args.num_questions,
        int(summary.get("repeat_times", -1)) == args.num_repeats,
        int(summary.get("top_k", -1)) == args.top_k,
        int(summary.get("adv_per_query", -1)) == args.adv_per_query,
        str(summary.get("random_targets")) == str(args.random_targets),
        str(summary.get("reuse_targets_per_repeat")) == str(args.reuse_targets_per_repeat),
    ]
    return all(checks)


def print_table(results_summary, attack_order, datasets):
    print("\n" + "=" * 110)
    print("ASR COMPARISON TABLE")
    print("=" * 110)

    header = f"{'Attack Method':<20} | " + " | ".join(f"{d.upper():<24}" for d in datasets)
    print(header)
    print("-" * len(header))

    for method_name, _ in attack_order:
        row = f"{method_name:<20} | "
        chunks = []
        for dataset in datasets:
            stats = results_summary.get(dataset, {}).get(method_name)
            if stats is None:
                chunks.append(f"{'N/A':<24}")
            else:
                chunks.append(f"{stats['mean']:.2%} +/- {stats['std']:.2%}")
        row += " | ".join(chunks)
        print(row)


def main():
    args = parse_args()
    resume = str2bool(args.resume)

    os.makedirs(args.output_dir, exist_ok=True)
    runs_dir = os.path.join(args.output_dir, "runs")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    results_summary = defaultdict(dict)

    total_runs = len(args.datasets) * len(ATTACK_CONFIGS)
    run_counter = 0

    print("\n" + "=" * 90)
    print("ASR EVALUATION (OPTIMIZED)")
    print("=" * 90)
    print(f"Datasets: {args.datasets}")
    print(f"Attacks: {[name for name, _ in ATTACK_CONFIGS]}")
    print(f"Questions per dataset: {args.num_questions}")
    print(f"Repeats per method: {args.num_repeats}")
    print(f"Total runs: {total_runs} (one run per dataset+method)")
    print("=" * 90)

    for dataset in args.datasets:
        print(f"\n{'#' * 90}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'#' * 90}")

        for method_name, attack_method in ATTACK_CONFIGS:
            run_counter += 1
            run_summary_path = os.path.join(runs_dir, f"asr_{dataset}_{method_name}.json")

            print(f"\n[{run_counter}/{total_runs}] Method: {method_name}")

            run_summary = None
            if resume and os.path.exists(run_summary_path):
                with open(run_summary_path, "r", encoding="utf-8") as f:
                    candidate_summary = json.load(f)
                if summary_matches_args(candidate_summary, args, dataset, attack_method):
                    run_summary = candidate_summary
                    print("Using existing run summary (resume=True).")
                else:
                    print("Existing summary parameters do not match current run; re-running.")

            if run_summary is None:
                completed, elapsed, run_name = run_single_experiment(
                    args=args,
                    dataset=dataset,
                    method_name=method_name,
                    attack_method=attack_method,
                    run_summary_path=run_summary_path,
                )

                log_prefix = os.path.join(logs_dir, run_name)
                with open(f"{log_prefix}.stdout.log", "w", encoding="utf-8") as f:
                    f.write(completed.stdout or "")
                with open(f"{log_prefix}.stderr.log", "w", encoding="utf-8") as f:
                    f.write(completed.stderr or "")

                if completed.returncode != 0:
                    print(f"Run failed (exit={completed.returncode}) after {elapsed / 60:.1f} min.")
                    print(f"See logs: {log_prefix}.stdout.log / {log_prefix}.stderr.log")
                    raise RuntimeError(f"Failed run for dataset={dataset}, method={method_name}")

                if not os.path.exists(run_summary_path):
                    raise FileNotFoundError(f"Expected summary missing: {run_summary_path}")

                with open(run_summary_path, "r", encoding="utf-8") as f:
                    run_summary = json.load(f)

                print(f"Completed in {elapsed / 60:.1f} min.")

            asr_values = run_summary.get("asr_per_repeat", [])
            if not asr_values:
                raise ValueError(f"No ASR values found in {run_summary_path}")

            asr_values_np = np.array(asr_values, dtype=float)
            stats = {
                "values": asr_values_np.tolist(),
                "mean": float(np.mean(asr_values_np)),
                "std": float(np.std(asr_values_np)),
                "min": float(np.min(asr_values_np)),
                "max": float(np.max(asr_values_np)),
            }
            results_summary[dataset][method_name] = stats

            print(
                f"ASR: {stats['mean']:.2%} +/- {stats['std']:.2%} "
                f"(range {stats['min']:.2%} to {stats['max']:.2%})"
            )

    summary_path = os.path.join(args.output_dir, "asr_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print_table(results_summary, ATTACK_CONFIGS, args.datasets)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Run summaries: {runs_dir}")
    print(f"Execution logs: {logs_dir}")


if __name__ == "__main__":
    main()
