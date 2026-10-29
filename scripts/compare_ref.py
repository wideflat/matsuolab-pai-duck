"""Compare two reference-motion sets (polynomial_coefficients.pkl) and report foot lift from the recordings.

Loads each pkl with Open_Duck_Playground's own PolyReferenceMotion (numpy version, as used by the inference
loop), evaluates every grid motion at every step of one gait period, and reports the largest absolute
difference per block (joint pos, joint vel, contacts, base lin vel, base ang vel). With --rec_a/--rec_b,
also reports the toe height range (max - min of toe z over the fitted period) from the JSON recordings.

Usage: PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/compare_ref.py --a <pkl> --b <pkl> \
           [--rec_a <dir> --rec_b <dir>]
Guarantees: raises if the motion grids or gait periods differ.
"""
import argparse
import json
from glob import glob

import numpy as np
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

BLOCKS = {"joint_pos": (0, 16), "joint_vel": (16, 32), "contacts": (32, 34), "lin_vel": (34, 37), "ang_vel": (37, 40)}


def toe_range(rec_dir):
    out = []
    for f in glob(f"{rec_dir}/*.json"):
        with open(f) as fh:
            d = json.load(fh)
        off, fps, period = d["Frame_offset"][0], d["FPS"], d["Placo"]["period"]
        s = int(d["Placo"]["startend_double_support_ratio"] * fps)
        y = np.array(d["Frames"])[s:s + int(period * fps)]
        for k in ("left_toe_pos", "right_toe_pos"):
            z = y[:, off[k] + 2]
            out.append(z.max() - z.min())
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--rec_a")
    p.add_argument("--rec_b")
    args = p.parse_args()
    A, B = PolyReferenceMotion(args.a), PolyReferenceMotion(args.b)
    for attr in ("dxs", "dys", "dthetas", "nb_steps_in_period"):
        if not np.allclose(getattr(A, attr), getattr(B, attr)):
            raise ValueError(f"{attr} differs: {getattr(A, attr)} vs {getattr(B, attr)}")
    worst = {k: 0.0 for k in BLOCKS}
    for dx in A.dxs:
        for dy in A.dys:
            for dth in A.dthetas:
                for i in range(A.nb_steps_in_period):
                    diff = np.abs(np.asarray(A.get_reference_motion(dx, dy, dth, i))
                                  - np.asarray(B.get_reference_motion(dx, dy, dth, i)))
                    for name, (lo, hi) in BLOCKS.items():
                        worst[name] = max(worst[name], float(diff[lo:hi].max()))
    n = len(A.dxs) * len(A.dys) * len(A.dthetas)
    print(f"{n} motions x {A.nb_steps_in_period} steps; max |a-b| per block:", {k: round(v, 4) for k, v in worst.items()})
    for name, d in (("a", args.rec_a), ("b", args.rec_b)):
        if d:
            r = toe_range(d)
            print(f"toe z range over one period ({name}): median {np.median(r)*100:.2f} cm, max {r.max()*100:.2f} cm")


if __name__ == "__main__":
    main()
