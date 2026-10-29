"""Print joints, ranges, actuators, masses and contact setup of an MJCF; optionally render a still.

Usage: .venv/bin/python scripts/inspect_mjcf.py <scene.xml> [--png out.png]
Guarantees: read-only on the model; prints a markdown table suitable for docs/notes.md.
"""
import argparse

import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("xml")
parser.add_argument("--png")
args = parser.parse_args()

m = mujoco.MjModel.from_xml_path(args.xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print(f"nbody={m.nbody} njnt={m.njnt} nu={m.nu} ngeom={m.ngeom} nq={m.nq} nv={m.nv}")
print(f"timestep={m.opt.timestep} gravity={m.opt.gravity.tolist()} total_mass={m.body_subtreemass[0]:.3f} kg")
print()
print("| joint | type | range (rad) | range (deg) | actuator | kp | forcerange |")
print("|---|---|---|---|---|---|---|")
act_by_jnt = {m.actuator_trnid[i, 0]: i for i in range(m.nu)}
types = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
for j in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    t = types[m.jnt_type[j]]
    if t == "free":
        print(f"| {name} | free | - | - | - | - | - |")
        continue
    lo, hi = m.jnt_range[j]
    a = act_by_jnt.get(j)
    if a is None:
        act, kp, fr = "-", "-", "-"
    else:
        act = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        kp = f"{m.actuator_gainprm[a, 0]:g}"
        fr = f"{m.actuator_forcerange[a].tolist()}"
    print(f"| {name} | {t} | [{lo:.2f}, {hi:.2f}] | [{np.degrees(lo):.0f}, {np.degrees(hi):.0f}] | {act} | {kp} | {fr} |")
print()
print("contacts at initial pose:", d.ncon)
for i in range(d.ncon):
    c = d.contact[i]
    g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"  {g1} <-> {g2} dist={c.dist:.4f}")
print("sensors:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(m.nsensor)])
print("keyframes:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) for k in range(m.nkey)])

if args.png:
    from PIL import Image
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.azimuth, cam.elevation = 0.7, 135, -15
    cam.lookat[:] = [0, 0, 0.18]
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(args.png)
    print("saved", args.png)
