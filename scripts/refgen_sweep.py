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

Recordings that start with a knee bent backwards (<= 0 rad at the first frame, a mirrored IK solution kept
for the whole motion) are regenerated with low parallelism, up to RETRY_ROUNDS times.

Guarantees: fails if any grid point fails to generate or any recording still has a knee bent backwards;
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
RETRY_ROUNDS = 5
RETRY_JOBS = 2


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
    used = out / "preset_used.json"
    if used.exists() and json.loads(used.read_text()) != preset:
        raise ValueError(f"{out} holds recordings from a different preset; use a new --out")
    used.write_text(json.dumps(preset, indent=2))

    points = grid(props)[:a.limit]
    cmds = []
    for i, (dx, dy, dth) in enumerate(points):
        pr = dict(preset, dx=dx, dy=dy, dtheta=dth)
        path = presets / f"{i}.json"
        path.write_text(json.dumps(pr, indent=4))
        cmds.append((i, [sys.executable, str(work / "open_duck_reference_motion_generator" / "gait_generator.py"),
                         "--duck", DUCK, "--preset", str(path), "--name", str(i), "--output_dir", str(rec)]))

    # One thread per process, to avoid oversubscribing the CPU (did not by itself fix the knee flips below).
    env =dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(item):
        i, cmd = item
        with open(logs / f"{i}.log", "w") as f:
            return i, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=work, env=env).returncode

    def generate(items, jobs):
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            failed = [i for i, rc in ex.map(run, items) if rc != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} grid points failed, e.g. {failed[:5]} (see {logs})")

    def flipped():
        bad = []
        for f in sorted(rec.glob("*.json")):
            d = json.loads(f.read_text())
            jp = np.array(d["Frames"])[0, d["Frame_offset"][0]["joints_pos"]:]
            if jp[KNEES].min() <= 0:  # first frame only: the two fastest grid corners cross 0 mid-motion
                bad.append(f)         # in the official pkl too, which is not this failure
        return bad

    have = {int(f.name.split("_")[0]) for f in rec.glob("*.json")}  # resume: same preset (checked above)
    generate([c for c in cmds if c[0] not in have], a.jobs)
    # Under parallel load the IK sometimes starts on the mirrored solution (one knee bent backwards, its hip
    # and ankle compensating) and keeps it for the whole motion; 23-30 of 240 per run, single-threaded or
    # not. The same grid point regenerated with little load comes out bent forwards, so retry those.
    for rnd in range(RETRY_ROUNDS):
        bad = flipped()
        print(f"round {rnd}: {len(bad)} recordings with a knee bent backwards")
        if not bad:
            break
        idx = {int(f.name.split("_")[0]) for f in bad}
        for f in bad:
            f.unlink()
        generate([c for c in cmds if c[0] in idx], RETRY_JOBS)
    n_json = len(list(rec.glob("*.json")))
    if n_json != len(points):
        raise RuntimeError(f"expected {len(points)} recordings, found {n_json}")
    bad = flipped()
    if bad:
        raise RuntimeError(f"{len(bad)} recordings still have a knee bent backwards, e.g. {[f.name for f in bad[:3]]}")
    subprocess.run([sys.executable, str(work / "scripts" / "fit_poly.py"), "--ref_motion", str(rec)],
                   cwd=out, check=True)
    print(f"done: {len(points)} motions -> {out / 'polynomial_coefficients.pkl'}")


if __name__ == "__main__":
    main()
