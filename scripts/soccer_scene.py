"""Soccer scene: the official flat-terrain scene plus a ball, a goal, and ball-only leg colliders.

The official robot model only collides through its foot soles, so a ball would pass through the legs.
build_model() adds massless capsules along each thigh and shin that collide with the ball only, then
attach() swaps the compiled model into a DuckRollout built on the official scene.

Collision bits (contact if contype1 & conaffinity2 or contype2 & conaffinity1):
  floor 1/0, feet 1/1 (official); ball 1/3; leg capsules and goal 2/0 -> they touch the ball only.

Guarantees: the robot's joints, actuators, sensors, qpos/qvel addresses and body masses are identical to
the official model (asserted), so the policy sees exactly the same observation layout.
"""
import mujoco
import numpy as np
from etils import epath
from playground.open_duck_mini_v2 import base

from duck_rollout import MODEL

BALL_R = 0.12          # m, beach-ball size. Feet 11.7 cm apart, shins 12.4 cm, trunk underside 11.5 cm up:
                       # r=0.04/0.065 slip under the trunk and between the legs; r=0.08 wedges under the trunk's
                       # front edge, lifts the duck and it falls. At r=0.12 the centre is at mid-trunk height.
BALL_MASS = 0.05       # kg
BALL_DAMPING = 0.01    # N s/m on the ball's free joint, so it does not roll forever (floor has no rolling friction)
LEG_R = 0.022          # m, radius of the ball-only leg capsules
GOAL_X = 1.5           # m, goal line
GOAL_HALF_W = 0.3      # m, half the distance between the posts (ball diameter 0.24)
GOAL_DEPTH = 0.35      # m, deep enough for the whole ball to cross the line
LEG_SEGMENTS = [("left_hip_pitch", "left_knee"), ("left_knee", "left_ankle"),
                ("right_hip_pitch", "right_knee"), ("right_knee", "right_ankle")]


def _official():
    return mujoco.MjModel.from_xml_string(epath.Path(MODEL).read_text(), assets=base.get_assets())


def build_model():
    spec = mujoco.MjSpec.from_file(MODEL)  # from_string cannot resolve the <include>; same files on disk
    ref = _official()
    d = mujoco.MjData(ref)
    d.qpos[:] = ref.keyframe("home").qpos
    mujoco.mj_forward(ref, d)

    # Leg capsules: the segment between consecutive joint anchors, attached to the body that owns the first
    # joint (the thigh owns hip_pitch, the shin owns knee), expressed in that body's frame at the home pose.
    for j0, j1 in LEG_SEGMENTS:
        body_id = ref.jnt_bodyid[ref.joint(j0).id]
        pos, rot = d.xpos[body_id], d.xmat[body_id].reshape(3, 3)
        a, b = (rot.T @ (d.xanchor[ref.joint(j).id] - pos) for j in (j0, j1))
        spec.body(ref.body(body_id).name).add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.concatenate([a, b]), size=[LEG_R, 0, 0],
            contype=2, conaffinity=0, density=0, group=3, rgba=[0, 0, 0, 0])

    # Trunk box: the trunk mesh's world bounding box at the home pose, in the trunk body's frame.
    tb = ref.body("trunk_assembly").id
    pos, rot = d.xpos[tb], d.xmat[tb].reshape(3, 3)
    pts = np.vstack([_mesh_world(ref, d, g) for g in np.where(ref.geom_bodyid == tb)[0]
                     if ref.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
    lo, hi = pts.min(0), pts.max(0)
    spec.body("trunk_assembly").add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, pos=rot.T @ ((lo + hi) / 2 - pos), quat=_mat2quat(rot.T),
        size=(hi - lo) / 2, contype=2, conaffinity=0, density=0, group=3, rgba=[0, 0, 0, 0])

    ball = spec.worldbody.add_body(name="ball", pos=[0.4, 0, BALL_R])
    ball.add_joint(name="ball", type=mujoco.mjtJoint.mjJNT_FREE, damping=BALL_DAMPING)
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_MASS,
                  contype=1, conaffinity=3, rgba=[0.95, 0.45, 0.1, 1])

    post = dict(contype=2, conaffinity=0, rgba=[0.15, 0.15, 0.15, 1])
    w = spec.worldbody
    for s in (1, -1):
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[GOAL_X, s * GOAL_HALF_W, 0.15], size=[0.012, 0.15, 0], **post)
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[GOAL_X + GOAL_DEPTH / 2, s * (GOAL_HALF_W + 0.005), 0.075],
                   size=[GOAL_DEPTH / 2, 0.005, 0.075], **post)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[GOAL_X + GOAL_DEPTH, 0, 0.075],
               size=[0.005, GOAL_HALF_W + 0.01, 0.075], **post)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[GOAL_X, 0, 0.0005], size=[0.004, GOAL_HALF_W, 0.0005],
               contype=0, conaffinity=0, rgba=[0.85, 0.1, 0.1, 1])

    key = spec.key("home")
    key.qpos = list(ref.keyframe("home").qpos) + [0.4, 0, BALL_R, 1, 0, 0, 0]
    model = spec.compile()
    _check_same_robot(ref, model)
    return model


def _mesh_world(m, d, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _mat2quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def _check_same_robot(ref, m):
    n = ref.njnt
    assert m.njnt == n + 1 and m.nu == ref.nu and m.nsensor == ref.nsensor, "robot structure changed"
    assert [m.joint(i).name for i in range(n)] == [ref.joint(i).name for i in range(n)], "joint order changed"
    assert np.array_equal(m.jnt_qposadr[:n], ref.jnt_qposadr) and np.array_equal(m.jnt_dofadr[:n], ref.jnt_dofadr)
    assert np.array_equal(m.sensor_adr, ref.sensor_adr) and np.array_equal(m.actuator_trnid, ref.actuator_trnid)
    assert np.allclose(m.body_mass[:ref.nbody], ref.body_mass), "leg capsules changed the robot mass"
    assert np.allclose(m.keyframe("home").ctrl, ref.keyframe("home").ctrl)


def attach(rollout, model, ball_xy):
    """Swap `model` into a DuckRollout built on the official scene and place the ball at ball_xy."""
    inf = rollout.inf
    model.opt.timestep = inf.sim_dt
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe("home").qpos
    data.qpos[-7:-5] = ball_xy
    data.ctrl[:] = inf.default_actuator
    mujoco.mj_forward(model, data)
    inf.model, inf.data = model, data
    rollout.model, rollout.data = model, data
    return rollout


def feet_xy(model, data):
    """Midpoint of the two foot bodies (x, y). Drifts ~9 cm in a 6 s in-place turn vs 15-17 cm for the base."""
    return data.xpos[[model.body("foot_assembly").id, model.body("foot_assembly_2").id], :2].mean(0)


def ball_xy(data):
    return data.qpos[-7:-5].copy()


def is_goal(data):
    """The whole ball is past the goal line and between the posts."""
    x, y = ball_xy(data)
    return bool(x - BALL_R > GOAL_X and abs(y) < GOAL_HALF_W)
