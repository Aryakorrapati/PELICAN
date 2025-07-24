#!/usr/bin/env python3
"""
run_double_descent.py
-------------------------------------------------
Automates a capacity‑sweep (for double descent):

 • Generates scaled copies of a base YAML (×K channels).
 • Trains one model per YAML (calls train_pelican_classifier.py).
 • Parses the log files to find the best *validation* loss.
 • Saves a double‑descent plot   log/double_descent_capacity.png
"""

import os, sys, shutil, subprocess, re, yaml, math, time, json
from pathlib import Path
import matplotlib.pyplot as plt

# ----------- USER‑TUNABLE SECTION -------------------------------------------
BASE_YAML      = "./config/60k.yaml"      # seed config you already use
DATA_DIR       = "./data/sample_data/run12"
TARGET         = "is_signal"
BATCH_SIZE     = 64
NUM_EPOCH      = 100                      # trainer’s --num-epoch
SCALE_FACTORS  = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # widths to try
PYTHON_BIN     = "python3"                # or full path
TRAIN_SCRIPT   = "train_pelican_classifier.py"
LOG_DIR        = Path("./log")
MODEL_DIR      = Path("./model")
# ---------------------------------------------------------------------------

def scale_channels(channels, K):
    """Multiply every int in a (possibly nested) list by K, round to int>=1"""
    if isinstance(channels, list):
        return [scale_channels(c, K) for c in channels]
    elif isinstance(channels, int):
        return max(1, int(round(channels * K)))
    else:
        return channels  # leave non‑ints unchanged

def make_scaled_yaml(base_yaml, K):
    """Return path to a scaled copy of base_yaml (creates file if missing)."""
    out_name = Path(base_yaml).with_suffix("").name + f"_{K:.2f}x.yaml"
    out_path = Path("./config") / out_name
    if out_path.exists():                 # reuse if already built
        return str(out_path)
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    for key in ["num_channels_m", "num_channels_2to2",
                "num_channels_m_out", "num_channels_out"]:
        if key in cfg:
            cfg[key] = scale_channels(cfg[key], K)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(out_path)

def run_one(K):
    """Launch one training job for scale factor K and return log‑file path."""
    yaml_path = make_scaled_yaml(BASE_YAML, K)
    prefix    = f"capacity_{K:.2f}x"
    log_path  = LOG_DIR / f"{prefix}.log"

    cmd = [
        PYTHON_BIN, TRAIN_SCRIPT,
        f"--datadir={DATA_DIR}",
        f"--yaml={yaml_path}",
        f"--target={TARGET}",
        f"--batch-size={BATCH_SIZE}",
        f"--prefix={prefix}",
        f"--num-epoch={NUM_EPOCH}",
        "--quiet"                           # cut stdout spam – still logs
    ]

    print("▶", " ".join(cmd))
    with open(log_path, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return log_path

VAL_PAT = re.compile(r"(val|valid|validation)[^\d]*(loss|L)[^\d]*([0-9]+\.[0-9]+)", re.I)

def best_val_loss(log_path):
    """Scan a trainer log and return the lowest validation loss it ever wrote."""
    best = math.inf
    with open(log_path) as f:
        for line in f:
            m = VAL_PAT.search(line)
            if m:
                v = float(m.group(3))
                best = min(best, v)
    return best if best < math.inf else None

def main():
    LOG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    results = []  # list of (K, best_val)

    for K in SCALE_FACTORS:
        try:
            logp = run_one(K)
            val  = best_val_loss(logp)
            print(f"  ↳  K={K:.2f} best val‑loss={val}")
            results.append((K,val))
        except subprocess.CalledProcessError:
            print(f"  ✖ training failed for K={K:.2f}")
    
    # ---------- plot ----------
    results = sorted([r for r in results if r[1] is not None])
    if not results:
        print("No valid results found ‑ nothing to plot.")
        return

    xs, ys = zip(*results)
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, "o-")
    plt.xlabel("width scale K")
    plt.ylabel("best validation loss ↓")
    plt.title("Double‑descent capacity sweep")
    plt.grid(alpha=.3)
    out_png = LOG_DIR / "double_descent_capacity.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print("✅  figure written to", out_png)

if __name__ == "__main__":
    main()
