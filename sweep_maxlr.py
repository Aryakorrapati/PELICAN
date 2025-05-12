import subprocess, re

TRAIN_SCRIPT = "train_pelican_classifier.py"
COMMON_ARGS = [
    "--datadir", "./data/sample_data/run12",
    "--yaml",    "./config/48k.yaml",
    "--batch-size", "64",
    "--prefix",     "swan"
]
MAX_LRS = [0.01, 0.008, 0.0065, 0.005, 0.004]

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
best_lr  = None
best_auc = None

for lr in MAX_LRS:
    cmd = ["python3", TRAIN_SCRIPT] + COMMON_ARGS + ["--max_lr", str(lr)]
    print(f"\n→ Running max_lr = {lr}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    acc, auc = parse_metrics(out)
    if acc is None:
        print("  Couldn't parse accuracy from output!")
        print(out)
        continue

    print(f"  max_lr={lr} → accuracy={acc:.4f}, AUC={auc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_auc = auc
        best_lr  = lr

if best_lr is not None:
    print(f"\n Best max_lr = {best_lr}  →  accuracy = {best_acc:.4f}, AUC = {best_auc:.4f}")
else:
    print("\n  No runs succeeded, check above for errors.")
