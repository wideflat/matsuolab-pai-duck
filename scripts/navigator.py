"""Navigator: a high-level controller that steers a trained walking policy to target points (no training).

Each control period it measures the distance d and heading error alpha to the current target and sets the
joystick command: wz = clip(K * alpha, +-WZ_MAX), vx = VX when |alpha| < ALIGN else 0 (turn in place), vy = 0.
The policy itself is unchanged; only its command input is driven.

Guarantees: deterministic for a deterministic rollout; reads the true base pose from the simulator
(simulation only -- a real robot would need its own localization); never commands outside the trained
ranges (|vx| <= 0.15, |wz| <= 1.0).
"""
import numpy as np

K = 2.5                    # heading gain, (rad/s) per rad
WZ_MAX = 1.0               # rad/s, edge of the trained wz range
VX = 0.15                  # m/s, edge of the trained forward range
ALIGN = np.radians(20.0)   # walk forward only when the heading error is below this
ARRIVE = 0.05              # m, target counts as reached inside this radius


def yaw(data):
    q = data.qpos[3:7]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def command_to(data, target, pos=None):
    """Return (vx, vy, wz, d, alpha) to steer at `target` (x, y in world metres).

    `pos` is the point measured against the target (default: the base origin); heading is the base yaw.
    """
    pos = data.qpos[0:2] if pos is None else pos
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    d = float(np.hypot(dx, dy))
    alpha = wrap(np.arctan2(dy, dx) - yaw(data))
    wz = float(np.clip(K * alpha, -WZ_MAX, WZ_MAX))
    vx = VX if abs(alpha) < ALIGN else 0.0
    return vx, 0.0, wz, d, alpha


def go_to(rollout, target, max_seconds):
    """Drive `rollout` (a DuckRollout) until it is within ARRIVE of target, it falls, or max_seconds pass.

    Returns dict(reached, t, d_final, fell_at, path) where path is the base xy sampled every control period.
    """
    r, dec = rollout, rollout.inf.decimation
    t_end = r.t + max_seconds
    path, d = [], np.inf
    while r.t < t_end and r.fell_at is None:
        if r.i % dec == 0:
            vx, vy, wz, d, _ = command_to(r.data, target)
            path.append((float(r.data.qpos[0]), float(r.data.qpos[1])))
            if d < ARRIVE:
                break
            r.inf.commands[0:3] = [vx, vy, wz]
        r.step()
    r.inf.commands[0:3] = [0.0, 0.0, 0.0]
    return {"reached": d < ARRIVE, "t": r.t, "d_final": d, "fell_at": r.fell_at, "path": np.array(path)}
