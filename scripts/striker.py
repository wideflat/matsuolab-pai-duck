"""Striker: picks the navigator's next target point so the duck pushes the ball into the goal (no training).

Work in the ball frame: u points from the ball to the aim point inside the goal, n is u turned 90 deg left,
s = (duck - ball).u (negative = behind the ball), e = (duck - ball).n (lateral offset).
  around   walk to a point beside the ball, LATERAL away on the side the duck is already on (level with the
           ball while still in front of it, else BEHIND back), so the body clears the ball.
  line_up  walk to the point BEHIND the ball on the ball->goal line.
  push     steer at a point PUSH_AHEAD past the ball centre: close enough to the centre that the ball stays
           on the front of the trunk, biased toward the goal so the line slowly straightens.
Switches use hysteresis so the mode does not chatter at a boundary:
  around -> line_up when s < -AROUND_DONE;  line_up -> around when s > -AROUND_START
  line_up -> push   within LINE_UP of the line-up point, or anywhere inside a CONE_DEG cone behind the ball;
  push -> line_up   when the ball is more than SLIP_DEG off the goal direction, or s > -AROUND_START,
                    judged only while walking (turning in place moves the feet a few cm and caused dithering).
Positions are the feet midpoint (soccer_scene.feet_xy): the duck turns about its feet, so that point moves
less in an in-place turn than the base origin. The duck pushes with the front of its trunk.

Guarantees: pure function of (duck position, ball position, mode, turning); all walking goes through
navigator.command_to, so commands stay inside the trained ranges.
"""
import numpy as np

from soccer_scene import GOAL_X

AIM = np.array([GOAL_X + 0.2, 0.0])   # m, aim point inside the goal
BEHIND = 0.3          # m, line-up distance behind the ball centre
AROUND_START = 0.12   # m, s above -AROUND_START means the duck is not behind the ball
AROUND_DONE = 0.25    # m, s below -AROUND_DONE means going around is finished
LATERAL = 0.45        # m, side clearance when going around (ball radius 0.12 + half duck width ~0.11 + margin)
LINE_UP = 0.10        # m, start pushing this close to the line-up point (turning in place shifts the base
                      # ~5 cm, so a tighter radius makes the bearing to the point swing and the duck dither)
PUSH_AHEAD = 0.08     # m, steer at this point past the ball centre while pushing
SLIP_DEG = 35.0       # deg, angle between duck->ball and ball->goal at which the push is abandoned
CONE_DEG = 20.0       # deg, already inside this cone behind the ball (and s < -AROUND_DONE): push at once


def frame(ball):
    u = AIM - ball
    u = u / np.linalg.norm(u)
    return u, np.array([-u[1], u[0]])


def target(duck_xy, ball, mode, turning):
    """Return (target_xy, new_mode) with mode in {"around", "line_up", "push"}.

    duck_xy is the feet midpoint. `turning` is True while the last command turned in place (vx = 0); the
    push is kept during such turns, because turning in place itself shifts the feet a few cm.
    """
    u, n = frame(ball)
    rel = duck_xy - ball
    s, e = float(rel @ u), float(rel @ n)
    behind = ball - BEHIND * u
    off = np.degrees(np.arctan2(abs(e), -s)) if s < 0 else 180.0

    if mode == "push":
        if turning or (off <= SLIP_DEG and s < -AROUND_START):
            return ball + PUSH_AHEAD * u, "push"
        mode = "line_up"
    if mode == "around" and s < -AROUND_DONE:
        mode = "line_up"
    if mode == "line_up" and s > -AROUND_START:
        mode = "around"

    if mode == "around":
        side = 1.0 if e >= 0 else -1.0
        along = 0.0 if s > 0 else -BEHIND
        return ball + side * LATERAL * n + along * u, "around"
    if np.linalg.norm(duck_xy - behind) < LINE_UP or (s < -AROUND_DONE and off < CONE_DEG):
        return ball + PUSH_AHEAD * u, "push"
    return behind, "line_up"
