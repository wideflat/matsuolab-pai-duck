"""Step-by-step rollout of an ONNX walking policy in MuJoCo (no viewer), shared by the replay tools.

Wraps MjInfer from external/Open_Duck_Playground (read-only) and re-implements only its control loop:
one call to step() advances one physics step (sim_dt) and runs the policy every `decimation` steps.

Guarantees: deterministic (no noise is added); with phase_rate=1 the control loop matches MjInfer.run()
at Playground b9be205; fell_at is the first sim time the base height drops below FALL_Z.
phase_rate scales how fast the imitation phase clock advances (1 = trained gait period 0.54 s); values
other than 1 are outside the training distribution.
"""
import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import USE_MOTOR_SPEED_LIMITS, MjInfer

MODEL = "external/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
REF = "external/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
FALL_Z = 0.08  # m, base height below which the robot counts as fallen


class DuckRollout:
    def __init__(self, onnx_path, command, model_path=MODEL, ref_path=REF, phase_rate=1.0):
        vx, vy, wz = command
        self.inf = MjInfer(model_path, ref_path, onnx_path, standing=False)
        self.inf.commands[0:3] = [vx, vy, wz]
        self.model, self.data = self.inf.model, self.inf.data
        self.phase_rate = phase_rate
        self.i = 0
        self.fell_at = None
        self.x0 = float(self.data.qpos[0])

    @property
    def t(self):
        return self.i * self.model.opt.timestep

    def step(self):
        inf, d = self.inf, self.data
        mujoco.mj_step(self.model, d)
        self.i += 1
        if self.i % inf.decimation == 0:
            inf.imitation_i = (inf.imitation_i + self.phase_rate) % inf.PRM.nb_steps_in_period
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
        if self.fell_at is None and d.qpos[2] < FALL_Z:
            self.fell_at = self.t

    @property
    def dx(self):
        return float(self.data.qpos[0]) - self.x0
