"""Train Open Duck Mini v2 (joystick env) with config overrides read from a JSON file in configs/.

external/Open_Duck_Playground is used read-only: this wraps its OpenDuckMiniV2Runner and only replaces
the train/eval environments with ones built from `config_overrides` (mujoco_playground's flattened-key
mechanism, e.g. {"reward_config.scales.tracking_lin_vel": 5.0}).

Usage (on the GPU machine, from ~/Open_Duck_Playground with its venv):
  python ~/train_variant.py --overrides ~/e004.json --task flat_terrain_backlash \
      --num_timesteps 50000000 --output_dir checkpoints_e004

--reference swaps the imitation target (Joystick hard-codes data/polynomial_coefficients.pkl) by replacing
env.PRM on both environments before training starts; PRM is only read inside reset/step.

Guarantees: an unknown override key raises (the base config is locked); before training, every
override is read back from both environments and must equal the requested value, else it raises;
a reference whose gait period differs from the official one raises.
"""
import argparse
import json

from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.open_duck_mini_v2 import joystick
from playground.open_duck_mini_v2.runner import OpenDuckMiniV2Runner


def read_back(config, flat_key):
    node = config
    for part in flat_key.split("."):
        node = node[part]
    return node


class VariantRunner(OpenDuckMiniV2Runner):
    def __init__(self, args, overrides):
        super().__init__(args)
        self.env = joystick.Joystick(task=args.task, config_overrides=overrides)
        self.eval_env = joystick.Joystick(task=args.task, config_overrides=overrides)
        for env in (self.env, self.eval_env):
            for key, want in overrides.items():
                got = read_back(env._config, key)
                if got != want:
                    raise ValueError(f"override {key}: requested {want}, env has {got}")
        if args.reference is not None:
            for env in (self.env, self.eval_env):
                prm = PolyReferenceMotion(args.reference)
                if prm.nb_steps_in_period != env.PRM.nb_steps_in_period:
                    raise ValueError(f"reference period {prm.nb_steps_in_period} steps != "
                                     f"official {env.PRM.nb_steps_in_period}")
                env.PRM = prm
            print("reference motion:", args.reference, "steps/period", prm.nb_steps_in_period)
        print("effective reward scales:", dict(self.env._config.reward_config.scales))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--overrides", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--num_timesteps", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--env", default="joystick")
    p.add_argument("--restore_checkpoint_path", default=None)
    p.add_argument("--reference", default=None, help="polynomial_coefficients.pkl to imitate (default: official)")
    args = p.parse_args()
    if args.env != "joystick":
        raise ValueError("train_variant.py only supports --env joystick")
    with open(args.overrides) as f:
        overrides = json.load(f)
    print("overrides:", overrides)
    VariantRunner(args, overrides).train()


if __name__ == "__main__":
    main()
