#!/usr/bin/env python3
import subprocess, re, sys
from datetime import datetime
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────
TRAIN_SCRIPT = "train_pelican_classifier.py"
COMMON_ARGS = [
    "--datadir", "./data/sample_data/run12",
    "--yaml",    "./config/48k.yaml",
    "--batch-size", "64",
    "--prefix",     "swan"
]
MAX_LRS = [0.01, 0.008, 0.0065, 0.005, 0.004]
LOG_FILE = os.path.expanduser("~/sweep_log.txt")
# ───────────────────────────────────────────────────────────────────────────

# Open log file for appending
log = open(LOG_FILE, "a", buffering=1)  # line-buffered

def log_print(*args, **kwargs):
    """Print to console *and* to our log file (with a newline)."""
    print(*args, **kwargs)
    print(*args, **kwargs, file=log)

def parse_metrics(output):
    """Extract accuracy and AUC from a block of text."""
    acc = None
    auc = None
    for line in output.splitlines():
        m = re.search(r"[Aa]ccuracy[:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: acc = float(m.group(1))
        m = re.search(r"[Aa][Uu][Cc][:=]\s*([0-9]*\.?[0-9]+)", line)
        if m: auc = float(m.group(1))
    return acc, auc

best_acc = -1.0
best_lr  = None
best_auc = None

log_print(f"\n=== SWEEP START {datetime.now():%Y-%m-%d %H:%M:%S} ===")

for lr in MAX_LRS:
    log_print(f"\n→ Starting run with max_lr = {lr}")
    cmd = ["python3", TRAIN_SCRIPT] + COMMON_ARGS + ["--max_lr", str(lr)]
    log_print("  Command:", " ".join(cmd))

    # Launch the training subprocess, merging stderr into stdout
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )

    # Stream output line by line
    full_output = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        log_print(line)
        full_output.append(line)
    proc.wait()

    # After it finishes, parse metrics
    out_block = "\n".join(full_output)
    acc, auc = parse_metrics(out_block)
    if acc is None:
        log_print("  ⚠️  ERROR: Couldn't find accuracy/AUC in output!")
        continue

    log_print(f"→ Completed max_lr={lr}:  accuracy={acc:.4f},  AUC={auc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_auc = auc
        best_lr  = lr

log_print("\n=== SWEEP COMPLETE ===")
if best_lr is not None:
    log_print(f"🏆 Best max_lr = {best_lr}  →  accuracy = {best_acc:.4f}, AUC = {best_auc:.4f}")
else:
    log_print("⚠️  No successful runs found.")
log_print(f"=== END {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")

log.close()
print(f"\n>> All output has been appended to {LOG_FILE}")
