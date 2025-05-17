#!/usr/bin/env python3
import subprocess, re, os, argparse
from datetime import datetime

def parse_metrics(output):
    """Extract the last occurrence of accuracy and AUC from the output text."""
    acc = None
    auc = None
    for line in output.splitlines():
        m = re.search(r"[Aa]ccuracy[:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: acc = float(m.group(1))
        m = re.search(r"[Aa][Uu][Cc][:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: auc = float(m.group(1))
    return acc, auc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annealing", choices=["cos","sin"], default="cos",
                   help="Which annealing to label the CSV (cos or sin)")
    args = p.parse_args()

    # ─── CONFIG ──────────────────────────────────────────────────────────
    TRAIN_SCRIPT = "train_pelican_classifier.py"
    COMMON_ARGS = [
        "--datadir", "./data/sample_data/run12",
        "--yaml",    "./config/48k.yaml",
        "--batch-size", "64",
        "--prefix",     "swan"
    ]
    MAX_LRS = [0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007]
    SWEEP_LOG = "sweep_log.txt"
    CSV_DIR   = "log"
    CSV_FILE  = os.path.join(CSV_DIR,
                  f"classifier.best.metrics.{args.annealing}.csv")
    # ──────────────────────────────────────────────────────────────────────

    # make sure the log directory exists
    os.makedirs(CSV_DIR, exist_ok=True)

    # open the sweep log
    log = open(SWEEP_LOG, "a", buffering=1)
    def log_print(*a, **kw):
        print(*a, **kw)
        print(*a, **kw, file=log)

    log_print(f"\n=== SWEEP START {datetime.now():%Y-%m-%d %H:%M:%S} "
              f"(annealing={args.annealing}) ===")

    results = []
    best_acc = -1.0
    best_lr  = None
    best_auc = None

    for lr in MAX_LRS:
        log_print(f"\n→ Starting run with max_lr = {lr}")
        cmd = ["python3", TRAIN_SCRIPT] + COMMON_ARGS + ["--max_lr", str(lr)]
        log_print("  Command:", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        full_output = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            log_print(line)
            full_output.append(line)
        proc.wait()

        out_block = "\n".join(full_output)
        acc, auc = parse_metrics(out_block)
        if acc is None:
            log_print("  ⚠️  ERROR: Couldn't find accuracy/AUC in output!")
            continue

        log_print(f"→ Completed max_lr={lr}:  accuracy={acc:.4f},  AUC={auc:.4f}")
        results.append((lr, acc, auc))
        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_lr  = lr

    # write out the CSV
    with open(CSV_FILE, "w") as f:
        f.write("max_lr,accuracy,auc\n")
        for lr,acc,auc in results:
            f.write(f"{lr:.6f},{acc:.6f},{auc:.6f}\n")

    log_print("\n=== SWEEP COMPLETE ===")
    if best_lr is not None:
        log_print(f"🏆 Best max_lr = {best_lr:.6f}  →  accuracy = {best_acc:.4f}, "
                  f"AUC = {best_auc:.4f}")
    else:
        log_print("⚠️  No successful runs found.")
    log_print(f"=== END {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    log.close()

    print(f"\n>> Logged console output to {SWEEP_LOG}")
    print(f">> Wrote per-lr best metrics to {CSV_FILE}")

if __name__ == "__main__":
    main()
