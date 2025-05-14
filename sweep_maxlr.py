#!/usr/bin/env python3
import subprocess, re, sys
from datetime import datetime

# --- CONFIGURE THESE ---
TRAIN_SCRIPT = "train_pelican_classifier.py"
COMMON_ARGS = [
    "--datadir", "./data/sample_data/run12",
    "--yaml",    "./config/48k.yaml",
    "--batch-size", "64",
    "--prefix",     "swan"
]
MAX_LRS = [0.01, 0.008, 0.0065, 0.005, 0.004]
# ------------------------

# Open a timestamped log file
log_filename = f"sweep_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
log = open(log_filename, "w")

def log_print(*args, **kwargs):
    """Print to stdout and also write to our log file."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

def parse_metrics(output):
    acc = None
    auc = None
    for line in output.splitlines():
        m = re.search(r"[Aa]ccuracy[:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: acc = float(m.group(1))
        m = re.search(r"[Aa][Uu][Cc][:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: auc = float(m.group(1))
    return acc, auc

best_acc = -1.0
best_lr = None
best_auc = None

log_print(f"Starting sweep at {datetime.now():%Y-%m-%d %H:%M:%S}")
for lr in MAX_LRS:
    log_print(f"\n→ Running max_lr = {lr}")
    cmd = ["python3", TRAIN_SCRIPT] + COMMON_ARGS + ["--max_lr", str(lr)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    # Write the full train script output to the log for debugging
    log_print("--- Begin training output ---")
    log_print(out)
    log_print("---- End training output ----")

    acc, auc = parse_metrics(out)
    if acc is None:
        log_print("  ⚠️  Couldn't parse accuracy from output!")
        continue

    log_print(f"  max_lr={lr} → accuracy={acc:.4f}, AUC={auc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_auc = auc
        best_lr = lr

if best_lr is not None:
    log_print(f"\n🏆 Best max_lr = {best_lr}  →  accuracy = {best_acc:.4f}, AUC = {best_auc:.4f}")
else:
    log_print("\n⚠️  No runs succeeded, check above for errors.")

log_print(f"Sweep completed at {datetime.now():%Y-%m-%d %H:%M:%S}")
log.close()

print(f"\n>> Logged full output to {log_filename}")
