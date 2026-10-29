"""Headless replay of an ONNX walking policy in MuJoCo (no viewer window).

The control loop lives in scripts/duck_rollout.py; this script reports numbers and can save a GIF
with a camera that follows the robot. For a fixed-camera side-by-side video use render_compare.py.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/replay_headless.py \
      -o external/published_policy/BEST_WALK_ONNX_2.onnx --vx 0.15 --seconds 10 --gif videos/replay.gif

Guarantees: prints seed-independent, deterministic results (no noise is added); never writes outside --gif.
"""
import argparse

import mujoco
from PIL import Image

from duck_rollout import MODEL, REF, DuckRollout

p = argparse.ArgumentParser()
p.add_argument("-o", "--onnx", required=True)
p.add_argument("--model", default=MODEL)
p.add_argument("--ref", default=REF)
p.add_argument("--vx", type=float, default=0.0)
p.add_argument("--vy", type=float, default=0.0)
p.add_argument("--wz", type=float, default=0.0)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--gif")
a = p.parse_args()

r = DuckRollout(a.onnx, (a.vx, a.vy, a.wz), a.model, a.ref)
m, d = r.model, r.data
steps = int(a.seconds / m.opt.timestep)
renderer = mujoco.Renderer(m, 240, 320) if a.gif else None
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.distance, cam.azimuth, cam.elevation = 0.9, 120, -15
frames = []
for _ in range(steps):
    r.step()
    if renderer is not None and r.i % int(1 / (30 * m.opt.timestep)) == 0:
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.15]
        renderer.update_scene(d, cam)
        frames.append(Image.fromarray(renderer.render()))

print(f"sim_seconds={a.seconds} base_z_final={d.qpos[2]:.3f} fell_at={r.fell_at}")
print(f"displacement_x={r.dx:.3f} m  mean_vx={r.dx / a.seconds:.3f} m/s (command vx={a.vx})  lateral_y={d.qpos[1]:.3f} m")
if frames:
    frames[0].save(a.gif, save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("saved", a.gif, len(frames), "frames")
