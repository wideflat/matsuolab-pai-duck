"""Render the walking reference motion (the imitation target) next to a trained policy, as an MP4.

Left panel: the reference motion from polynomial_coefficients.pkl played kinematically (no physics):
joint angles come from the reference, the base moves by integrating the reference base velocity, and
the base height is shifted each frame so the lower foot touches the floor. Right panel: an ONNX policy
simulated with physics (DuckRollout) under the same command. Same fixed camera as render_compare.py.

Reference layout (per frame, 40 dims; from Open_Duck_reference_motion_generator scripts/fit_poly.py):
  0:16 joint pos (16 incl. 2 antennas at 9, 10) | 16:32 joint vel | 32:34 foot contacts |
  34:37 base linear vel (world) | 37:40 base angular vel (world)

Usage (from the repo root):
  PYTHONPATH=external/Open_Duck_Playground .venv/bin/python scripts/render_reference.py \
      --onnx checkpoints/baseline_e003/2026_10_28_034228_50462720.onnx --label "E003 最終 (50.5M)" \
      --vx 0.15 --seconds 8 --out videos/reference_vs_e003.mp4 --poster videos/reference_vs_e003.png

Guarantees: deterministic; the reference frame must have 40 dims and map onto 14 actuated joints,
else it raises. Writes only --out and --poster.
"""
import argparse
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duck_rollout import REF, DuckRollout
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from render_compare import (CAM_AZ, CAM_DIST, CAM_EL, CAM_LOOKAT, COLORS, FONT, FPS, INK, MARK_X, SURFACE,
                            add_floor_marks, project)

PANEL_W, PANEL_H, HEADER = 640, 300, 44
FOOT_GEOMS = ("left_foot_bottom_tpu", "right_foot_bottom_tpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--vx", type=float, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--poster", required=True)
    a = p.parse_args()

    policy = DuckRollout(a.onnx, (a.vx, 0.0, 0.0))
    m = policy.model
    d_ref = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d_ref, m.key("home").id)
    home_qpos = d_ref.qpos.copy()
    mujoco.mj_forward(m, d_ref)
    foot_ids = [m.geom(g).id for g in FOOT_GEOMS]
    floor_z = min(d_ref.geom_xpos[i][2] for i in foot_ids)
    prm = PolyReferenceMotion(REF)
    ctrl_dt = 1.0 / prm.fps

    renderer = mujoco.Renderer(m, PANEL_H, PANEL_W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = CAM_LOOKAT, CAM_DIST, CAM_AZ, CAM_EL
    ticks = [(x, project(cam, m.vis.global_.fovy, (x, -0.12, 0.0), PANEL_W, PANEL_H)) for x in MARK_X]
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (22, 18, 13))
    W, H = 2 * PANEL_W, HEADER + PANEL_H

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", a.out], stdin=subprocess.PIPE)
    base_xy = home_qpos[0:2].copy()
    frame = None
    for n in range(int(a.seconds * FPS)):
        t = n / FPS
        ref = np.array(prm.get_reference_motion(a.vx, 0.0, 0.0, int(t / ctrl_dt) % prm.nb_steps_in_period))
        if ref.shape != (40,):
            raise ValueError(f"reference frame has shape {ref.shape}, expected (40,)")
        joints = np.concatenate([ref[0:9], ref[11:16]])
        if joints.shape != (14,):
            raise ValueError(f"reference joints {joints.shape}, expected (14,)")
        base_xy = base_xy + ref[34:36] / FPS
        d_ref.qpos[:] = home_qpos
        d_ref.qpos[0:2] = base_xy
        d_ref.qpos[7:21] = joints
        mujoco.mj_forward(m, d_ref)
        d_ref.qpos[2] -= min(d_ref.geom_xpos[i][2] for i in foot_ids) - floor_z
        mujoco.mj_forward(m, d_ref)
        policy.advance_to(t)

        frame = Image.new("RGB", (W, H), SURFACE)
        draw = ImageDraw.Draw(frame)
        draw.text((16, 10), f"手本(参照動作)と学習結果  指令: 前へ {a.vx} m/s", font=big, fill=INK)
        draw.text((W - 16, 10), f"t = {t:4.1f} s", font=big, fill=INK, anchor="ra")
        panels = [(d_ref, "手本: Placo の歩行生成器(物理なし)", base_xy[0] - home_qpos[0], COLORS[0]),
                  (policy.data, f"学習結果: {a.label}(物理あり)", policy.dx, COLORS[3])]
        for k, (data, title, dist, color) in enumerate(panels):
            renderer.update_scene(data, cam)
            add_floor_marks(renderer.scene, home_qpos[0])
            panel = Image.fromarray(renderer.render())
            pd = ImageDraw.Draw(panel)
            for x, (u, v) in ticks:
                if x * 10 % 5 == 0:
                    pd.text((u, v + 4), f"{x:.1f} m", font=small, fill="white", anchor="ma")
            pd.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=color, width=4)
            pd.rectangle([4, 4, 470, 34], fill=color)
            pd.text((12, 8), title, font=mid, fill="white")
            pd.text((PANEL_W - 12, 42), f"{dist:+.2f} m", font=big, fill="white", anchor="ra",
                    stroke_width=2, stroke_fill="black")
            frame.paste(panel, (k * PANEL_W, HEADER))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    frame.save(a.poster)
    print(f"wrote {a.out}; reference dx={base_xy[0] - home_qpos[0]:.3f} m, policy dx={policy.dx:.3f} m")


if __name__ == "__main__":
    main()
