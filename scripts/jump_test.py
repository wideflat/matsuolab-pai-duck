"""E008: how high can the duck jump? Hand-written crouch-and-extend joint targets, no policy, official scene.

Sequence: stand at the home pose 1 s -> crouch (knee +delta, hip and ankle -delta/2 each so the trunk stays
level) over 0.5 s -> extend at once (knee -ext, hip and ankle +ext/2, plus an optional ankle push) -> hold.
The right hip pitch axis is mirrored (home -0.63 left / +0.635 right), so its offsets flip sign.
Targets are clipped to each actuator's ctrlrange. Two target modes: "step" (targets jump instantly; the
actuator force limit 3.23 N m and joint damping set the speed) and "limited" (targets move at most 5.24 rad/s,
as in the deployment loop, playground mujoco_infer.py).

Measured per run: airtime = longest span with neither foot touching the floor; lift = during an airborne
span, the largest rise of the lower sole (lowest collision-mesh vertex) above its standing height; fell =
base below 0.08 m within 1.5 s of take-off. (The foot body origin sits at the ankle and rises when the foot
merely tilts, so it is not used.)

Usage: PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/jump_test.py --out docs/data/e008_jump.csv
Guarantees: deterministic; reads joint names and ranges from the model (asserts the expected joints exist).
"""
import argparse
import csv
import itertools
from pathlib import Path

import mujoco
import numpy as np
from etils import epath
from playground.open_duck_mini_v2 import base

from duck_rollout import FALL_Z, MODEL

DT = 0.002
MAX_VEL = 5.24  # rad/s, servo speed limit used by the deployment loop
STAND, CROUCH, AFTER = 1.0, 0.5, 1.5  # s
DELTAS = [0.0, 0.1, 0.2]              # rad, extra knee bend when crouching (knee limit 1.57, home ~1.37)
EXTS = [0.6, 0.9, 1.2, 1.5]           # rad, knee extension at take-off
PUSHES = [-0.3, 0.0, 0.3]             # rad, extra ankle rotation at take-off (sign of "toe push" not assumed)
LEGS = {"left": 1.0, "right": -1.0}   # hip pitch sign per side


def load():
    m = mujoco.MjModel.from_xml_string(epath.Path(MODEL).read_text(), assets=base.get_assets())
    m.opt.timestep = DT
    names = [m.actuator(i).name for i in range(m.nu)]
    for side in LEGS:
        for j in ("hip_pitch", "knee", "ankle"):
            assert f"{side}_{j}" in names, f"missing actuator {side}_{j}"
    return m, names


def pose(m, names, bend, ankle_extra=0.0):
    """Home ctrl with the knee bent by `bend` (negative = extend) and hip/ankle compensating."""
    q = m.keyframe("home").ctrl.copy()
    for side, s in LEGS.items():
        q[names.index(f"{side}_knee")] += bend
        q[names.index(f"{side}_hip_pitch")] -= s * bend / 2
        q[names.index(f"{side}_ankle")] += -bend / 2 + ankle_extra
    return np.clip(q, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def feet_on_floor(m, d, feet, floor):
    on = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if floor in (b1, b2):
            on.add(b2 if b1 == floor else b1)
    return [f in on for f in feet]


def sole_z(m, d, foot):
    """Lowest world z over the foot's collision mesh vertices (the sole), in m."""
    z = np.inf
    for g in np.where((m.geom_bodyid == foot) & (m.geom_contype > 0))[0]:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        z = min(z, float(np.min(v @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g][2])))
    return z


def run(m, names, delta, ext, push, mode):
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    d.ctrl[:] = m.keyframe("home").ctrl
    feet = [m.body("foot_assembly").id, m.body("foot_assembly_2").id]
    floor = m.body("floor").id
    home, low, high = m.keyframe("home").ctrl, pose(m, names, delta), pose(m, names, delta - ext, push)
    n_stand, n_crouch, n_after = (int(round(x / DT)) for x in (STAND, CROUCH, AFTER))
    for i in range(n_stand):
        mujoco.mj_step(m, d)
    foot_z0 = np.array([sole_z(m, d, f) for f in feet])
    for i in range(n_crouch):
        d.ctrl[:] = home + (low - home) * (i + 1) / n_crouch
        mujoco.mj_step(m, d)
    air, best_air, lift, best_lift, fell = 0, 0, 0.0, 0.0, False
    for i in range(n_after):
        if mode == "step":
            d.ctrl[:] = high
        else:
            d.ctrl[:] = d.ctrl + np.clip(high - d.ctrl, -MAX_VEL * DT, MAX_VEL * DT)
        mujoco.mj_step(m, d)
        if not any(feet_on_floor(m, d, feet, floor)):
            air += 1
            lift = max(lift, float(np.min([sole_z(m, d, f) for f in feet] - foot_z0)))
            if air > best_air:
                best_air, best_lift = air, max(best_lift, lift)
        else:
            air, lift = 0, 0.0
        fell |= d.qpos[2] < FALL_Z
    return {"airtime_s": best_air * DT, "lift_cm": best_lift * 100, "fell": fell}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m, names = load()
    rows = []
    for mode, delta, ext, push in itertools.product(("step", "limited"), DELTAS, EXTS, PUSHES):
        r = run(m, names, delta, ext, push, mode)
        rows.append({"mode": mode, "crouch_rad": delta, "extend_rad": ext, "ankle_push_rad": push,
                     "airtime_s": f"{r['airtime_s']:.3f}", "lift_cm": f"{r['lift_cm']:.2f}", "fell": r["fell"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for mode in ("step", "limited"):
        sub = [r for r in rows if r["mode"] == mode]
        best = max(sub, key=lambda r: float(r["lift_cm"]))
        print(mode, "best:", best, "| runs with airtime>0:", sum(float(r["airtime_s"]) > 0 for r in sub), "/", len(sub))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
