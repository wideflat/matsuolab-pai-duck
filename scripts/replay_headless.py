"""Headless replay of an ONNX walking policy in MuJoCo (no viewer window).

Reuses MjInfer from external/Open_Duck_Playground (read-only) and re-implements only its control loop
so it can run without a GUI, report numbers, and save a GIF.

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/replay_headless.py \
      -o external/published_policy/BEST_WALK_ONNX_2.onnx --vx 0.15 --seconds 10 --gif videos/replay.gif

Guarantees: prints seed-independent, deterministic results (no noise is added); never writes outside --gif.
"""
import argparse

import mujoco
import numpy as np
from PIL import Image

from playground.open_duck_mini_v2.mujoco_infer import USE_MOTOR_SPEED_LIMITS, MjInfer

p = argparse.ArgumentParser()
p.add_argument("-o", "--onnx", required=True)
p.add_argument("--model", default="external/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
p.add_argument("--ref", default="external/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
p.add_argument("--vx", type=float, default=0.0)
p.add_argument("--vy", type=float, default=0.0)
p.add_argument("--wz", type=float, default=0.0)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--gif")
a = p.parse_args()

inf = MjInfer(a.model, a.ref, a.onnx, standing=False)
inf.commands[0:3] = [a.vx, a.vy, a.wz]
m, d = inf.model, inf.data
steps = int(a.seconds / m.opt.timestep)
renderer = mujoco.Renderer(m, 240, 320) if a.gif else None
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.distance, cam.azimuth, cam.elevation = 0.9, 120, -15
frames, fell_at = [], None
x0 = d.qpos[0]
for i in range(1, steps + 1):
    mujoco.mj_step(m, d)
    if i % inf.decimation == 0:
        inf.imitation_i = (inf.imitation_i + 1.0) % inf.PRM.nb_steps_in_period
        ph = inf.imitation_i / inf.PRM.nb_steps_in_period * 2 * np.pi
        inf.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
        obs = inf.get_obs(d, inf.commands)
        action = inf.policy.infer(obs)
        inf.last_last_last_action = inf.last_last_action.copy()
        inf.last_last_action = inf.last_action.copy()
        inf.last_action = action.copy()
        inf.motor_targets = inf.default_actuator + action * inf.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            lim = inf.max_motor_velocity * (inf.sim_dt * inf.decimation)
            inf.motor_targets = np.clip(inf.motor_targets, inf.prev_motor_targets - lim, inf.prev_motor_targets + lim)
            inf.prev_motor_targets = inf.motor_targets.copy()
        d.ctrl = inf.motor_targets.copy()
    if fell_at is None and d.qpos[2] < 0.08:
        fell_at = i * m.opt.timestep
    if renderer is not None and i % int(1 / (30 * m.opt.timestep)) == 0:
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.15]
        renderer.update_scene(d, cam)
        frames.append(Image.fromarray(renderer.render()))

dx, dy = d.qpos[0] - x0, d.qpos[1]
print(f"sim_seconds={a.seconds} base_z_final={d.qpos[2]:.3f} fell_at={fell_at}")
print(f"displacement_x={dx:.3f} m  mean_vx={dx / a.seconds:.3f} m/s (command vx={a.vx})  lateral_y={dy:.3f} m")
if frames:
    frames[0].save(a.gif, save_all=True, append_images=frames[1:], duration=33, loop=0)
    print("saved", a.gif, len(frames), "frames")
