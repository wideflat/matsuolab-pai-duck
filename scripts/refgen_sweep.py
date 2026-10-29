"""Generate a reference-motion set (polynomial_coefficients.pkl) with Open_Duck_reference_motion_generator.

Runs inside the linux/amd64 container from configs/refgen.Dockerfile (placo 0.6.3 only ships a manylinux
x86_64 wheel). The generator repo is mounted read-only and copied to a scratch dir, so nothing in
external/ is written. For each point of the sweep grid in robots/open_duck_mini_v2/auto_gait.json it runs
gait_generator.py with the "medium" preset plus the overrides from --overrides (a JSON of preset keys),
then runs scripts/fit_poly.py in the output dir.

Unlike scripts/auto_waddle.py this keeps every grid point (auto_waddle deletes recordings whose speed falls
outside the preset's band; the 240-motion pkl shipped in Open_Duck_Playground has the full grid).

Usage (host, from the repo root):
  docker run --rm --platform linux/amd64 -v "$PWD/external/Open_Duck_reference_motion_generator:/gen:ro" \
      -v "$PWD/scripts:/scripts:ro" -v "$PWD/configs:/configs:ro" -v "$PWD/runs/refgen:/out" duck-refgen \
      python /scripts/refgen_sweep.py --overrides /configs/refgen_skip.json --out /out/skip --jobs 8

Guarantees: fails if any grid point fails to generate or any recording has a knee bent backwards (<= 0 rad);
every override key must already exist in the preset; each generator process runs single-threaded.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DUCK = "open_duck_mini_v2"
KNEES = [3, 14]  # left_knee, right_knee in the preset's joint order; both bend forward (positive) when walking


def grid(props):
    def arange(lo, hi, step):
        return np.arange(lo, hi + step, step)
    xs = arange(props["min_sweep_x"], props["max_sweep_x"], props["sweep_xy_granularity"])
    ys = arange(props["min_sweep_y"], props["max_sweep_y"], props["sweep_xy_granularity"])
    ts = arange(props["min_sweep_theta"], props["max_sweep_theta"], props["sweep_theta_granularity"])
    n = len(xs) * len(ys) * len(ts)
    # same index order as auto_waddle.py --sweep
    return [(round(xs[i % len(xs)], 2), round(ys[(i // len(xs)) % len(ys)], 2),
             round(ts[(i // (len(xs) * len(ys))) % len(ts)], 2)) for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--overrides", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="only the first N grid points (smoke test)")
    a = p.parse_args()

    work = Path("/tmp/gen")
    shutil.copytree("/gen", work, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".venv", ".git"))
    robot = work / "open_duck_reference_motion_generator" / "robots" / DUCK
    props = json.loads((robot / "auto_gait.json").read_text())
    preset = json.loads((robot / "placo_presets" / "medium.json").read_text())
    overrides = json.loads(Path(a.overrides).read_text())
    for k, v in overrides.items():
        if k not in preset:
            raise KeyError(f"override {k} is not a preset key")
        preset[k] = v

    out = Path(a.out)
    rec, presets, logs = out / "recordings", out / "presets", out / "logs"
    for d in (rec, presets, logs):
        d.mkdir(parents=True, exist_ok=True)
    (out / "preset_used.json").write_text(json.dumps(preset, indent=2))

    points = grid(props)[:a.limit]
    cmds = []
    for i, (dx, dy, dth) in enumerate(points):
        pr = dict(preset, dx=dx, dy=dy, dtheta=dth)
        path = presets / f"{i}.json"
        path.write_text(json.dumps(pr, indent=4))
        cmds.append((i, [sys.executable, str(work / "open_duck_reference_motion_generator" / "gait_generator.py"),
                         "--duck", DUCK, "--preset", str(path), "--name", str(i), "--output_dir", str(rec)]))

    # One thread per process: with 10 multithreaded processes the IK sometimes converged to the mirrored
    # (backward-bent) right knee on 30/240 motions; the same motion run alone was correct.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(item):
        i, cmd = item
        with open(logs / f"{i}.log", "w") as f:
            return i, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=work, env=env).returncode

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        failed = [i for i, rc in ex.map(run, cmds) if rc != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} grid points failed, e.g. {failed[:5]} (see {logs})")
    n_json = len(list(rec.glob("*.json")))
    if n_json != len(points):
        raise RuntimeError(f"expected {len(points)} recordings, found {n_json}")
    flipped = []
    for f in sorted(rec.glob("*.json")):
        d = json.loads(f.read_text())
        jp = np.array(d["Frames"])[:, d["Frame_offset"][0]["joints_pos"]:]
        if jp[:, KNEES].min() <= 0:
            flipped.append(f.name)
    if flipped:
        raise RuntimeError(f"{len(flipped)} recordings have a knee bent backwards, e.g. {flipped[:3]}")
    subprocess.run([sys.executable, str(work / "scripts" / "fit_poly.py"), "--ref_motion", str(rec)],
                   cwd=out, check=True)
    print(f"done: {len(points)} motions -> {out / 'polynomial_coefficients.pkl'}")


if __name__ == "__main__":
    main()
