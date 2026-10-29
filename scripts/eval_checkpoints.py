"""Roll out every ONNX checkpoint of a run and write one CSV row per checkpoint.

Checkpoint files are named <date>_<time>_<steps>.onnx by Open_Duck_Playground's runner; the step count
is parsed from the name. Rewards come from the runner's stdout log (lines "STEP: <n> reward: <r> ...").

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/eval_checkpoints.py \
      --run checkpoints/baseline_e003 --log checkpoints/baseline_e003/baseline.log \
      --vx 0.15 --seconds 10 --out docs/data/e003_checkpoints.csv

Guarantees: deterministic (no noise); every checkpoint must have a matching reward line, else it raises.
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

from duck_rollout import DuckRollout


def read_rewards(log_path):
    rewards = {}
    for line in Path(log_path).read_text().splitlines():
        hit = re.match(r"STEP: (\d+) reward: ([-0-9.eE]+)", line)
        if hit:
            rewards[int(hit.group(1))] = float(hit.group(2))
    return rewards


def list_checkpoints(run_dir):
    ckpts = []
    for f in Path(run_dir).glob("*.onnx"):
        ckpts.append((int(f.stem.rsplit("_", 1)[1]), f))
    return sorted(ckpts)


def sha8(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--vx", type=float, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rewards = read_rewards(a.log)
    rows = []
    for steps, onnx in list_checkpoints(a.run):
        r = DuckRollout(str(onnx), (a.vx, 0.0, 0.0))
        for _ in range(int(a.seconds / r.model.opt.timestep)):
            r.step()
        rows.append(dict(steps=steps, eval_reward=f"{rewards[steps]:.1f}", fell_at=r.fell_at or "",
                         dx_m=f"{r.dx:.3f}", mean_vx=f"{r.dx / a.seconds:.4f}", onnx_sha8=sha8(onnx)))
        print(rows[-1])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
