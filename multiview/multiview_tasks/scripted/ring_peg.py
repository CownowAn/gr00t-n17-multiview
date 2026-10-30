"""Scripted policy for RingOnPeg: carry the held ring over the peg, lower it onto the
peg, open the hand and withdraw.

Uses privileged state (ring and peg poses). The IK target is the right grip site,
kept in the rest hand orientation the episode starts in (the ring is held in it).
Every step the ring's offset in the hand is re-measured and the hand target is set so
that the ring (not the hand) follows the path:
    lift   straight up until the ring is LIFT_HEIGHT above the peg top
    carry  level, to above the peg
    settle, lower to LOWER_DEPTH below the peg top, release (open the hand),
    clear  move down and right: the open hand drops away from the crossbar resting
           on its fingers (moving only sideways dragged the ring off a 15 mm-clearance
           peg), then retract upward.
While settling and lowering, the horizontal offset between the ring center and the
peg axis is fed back into the hand target (the arm lags its commands by a few cm),
and the descent pauses whenever that offset exceeds ALIGN_FRACTION of the clearance.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.tasks.ring_peg import GRIP_SITE, HOLD_HAND, PEG_RADIUS

OPEN_HAND = np.zeros(6)
ORIENTATION_COST = 0.5
LIFT_HEIGHT = 0.06  # ring above the peg top while carrying (m)
# Ring center below the peg top when the hand opens (m). Lower would bring the wrist
# onto the platform; the released ring slides down the peg.
LOWER_DEPTH = 0.045
CLEAR_DOWN = 0.05
CLEAR_SIDE = 0.05
RETRACT_SIDE = 0.03
RETRACT_UP = 0.10
ALIGN_GAIN = 0.3  # fraction of the horizontal ring-peg offset added to the target per step
ALIGN_FRACTION = 0.5  # descend only while the offset is below this fraction of the clearance
MAX_CORRECTION = 0.05  # (m)
LOWER_STEPS = 50  # aligned steps needed to complete the descent
RELEASE_DEPTH = 0.025  # open the hand once the ring is this far below the peg top (m)
OPEN_STEPS = 15
OPEN_FLEXION = 0.2  # start moving away only once the fingers are this open (rad)

# (phase, duration in control steps at 20 Hz)
PHASES = (
    ("hold", 10),
    ("lift", 30),
    ("carry", 60),
    ("settle", 15),
    ("lower", 120),
    ("release", 40),
    ("clear", 20),
    ("retract", 25),
)
SEGMENTS = {  # phase -> (from waypoint, to waypoint)
    "hold": ("start", "start"),
    "lift": ("start", "lifted"),
    "carry": ("lifted", "above"),
    "settle": ("above", "above"),
    "lower": ("above", "lowered"),
    "release": ("lowered", "lowered"),
    "clear": ("lowered", "cleared"),
    "retract": ("cleared", "retracted"),
    "idle": ("retracted", "retracted"),
}
RING_GUIDED = ("lift", "carry", "settle", "lower")  # targets follow the measured ring


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class RingOnPegScriptedPolicy:
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    def reset(self, obs: dict, rng=None):
        env = self.gym_env.env
        model, data = env.sim.model._model, env.sim.data._data
        self.site = model.site(GRIP_SITE).id
        self.ik = ArmIK(model, "right", GRIP_SITE, frame_type="site", orientation_cost=ORIENTATION_COST)
        self.ik.reset(data.qpos)
        self.r_hand = Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3))
        base_rot = data.xmat[model.body("robot0_base").id].reshape(3, 3)
        self.right = -base_rot[:, 1]
        self.up = np.array([0.0, 0.0, 1.0])

        ring_center, _ = env.ring_pose()
        self.start_ring = ring_center
        self.waypoints = {"start": data.site_xpos[self.site].copy()}
        self._plan()
        self.bounds = np.cumsum([0] + [d for _, d in PHASES])
        self.correction = np.zeros(3)
        self.lower_progress = 0.0
        self.align_tol = ALIGN_FRACTION * (env.inner_radius - PEG_RADIUS)
        self.hold = {
            k: np.asarray(obs[f"state.{k}"], dtype=np.float64).copy()
            for k in ("left_arm", "left_hand", "waist")
        }
        self.t = 0

    def _plan(self):
        """Hand waypoints that put the ring (at its current offset in the hand) on the path."""
        env = self.gym_env.env
        data = env.sim.data._data
        ring_center, _ = env.ring_pose()
        rot = data.site_xmat[self.site].reshape(3, 3)
        ring_in_hand = rot.T @ (ring_center - data.site_xpos[self.site])
        to_site = lambda ring: ring - self.r_hand.apply(ring_in_hand)  # noqa: E731
        peg_top = env.peg_top()
        carry_z = peg_top[2] + LIFT_HEIGHT
        w = self.waypoints
        w["lifted"] = to_site(np.array([self.start_ring[0], self.start_ring[1], carry_z]))
        w["above"] = to_site(np.array([peg_top[0], peg_top[1], carry_z]))
        w["lowered"] = to_site(peg_top - LOWER_DEPTH * self.up)
        w["cleared"] = w["lowered"] - CLEAR_DOWN * self.up + CLEAR_SIDE * self.right
        w["retracted"] = w["cleared"] + RETRACT_SIDE * self.right + RETRACT_UP * self.up

    @property
    def total_steps(self) -> int:
        return int(self.bounds[-1])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        for (name, _), end in zip(PHASES, self.bounds[1:]):
            if t < end:
                return name
        return "idle"

    def _target(self, t: int):
        name = self.phase(t)
        if name == "idle":
            return self.waypoints["retracted"], name
        i = [n for n, _ in PHASES].index(name)
        s = (t - self.bounds[i]) / PHASES[i][1]
        a, b = (self.waypoints[k] for k in SEGMENTS[name])
        smooth = self.lower_progress if name == "lower" else min_jerk(s)
        return a + smooth * (b - a), name

    def _align(self, name):
        """Feed the horizontal ring-peg offset back; advance the descent only when aligned."""
        env = self.gym_env.env
        ring_center, _ = env.ring_pose()
        offset = env.peg_top() - ring_center
        offset[2] = 0.0
        self.correction = np.clip(self.correction + ALIGN_GAIN * offset, -MAX_CORRECTION, MAX_CORRECTION)
        if name == "lower" and np.linalg.norm(offset) < self.align_tol:
            self.lower_progress = min(1.0, self.lower_progress + 1.0 / LOWER_STEPS)

    def act(self) -> dict:
        name = self.phase()
        if name == "lower" and -self.gym_env.env._ring_offset()[1] > RELEASE_DEPTH and self.lower_progress >= 1.0:
            self.t = self.bounds[5]  # deep enough: skip the rest of the descent and release
            name = self.phase()
        if name in RING_GUIDED:
            self._plan()
        if name in ("settle", "lower"):
            self._align(name)
        if name == "release" and self.t >= self.bounds[5] + OPEN_STEPS:
            env = self.gym_env.env
            if env._released and env._finger_flexion() < OPEN_FLEXION:
                self.t = self.bounds[6]  # hand open: move away now (the rest of "release" is slack)
                name = self.phase()
        pos, _ = self._target(min(self.t, self.total_steps))
        if name in ("settle", "lower", "release"):
            pos = pos + self.correction
        target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self.r_hand.as_matrix()), pos)
        right_arm = self.ik.solve(target)
        if name in ("hold",) + RING_GUIDED:
            hand = HOLD_HAND
        elif name == "release":  # open gradually so the fingers do not flick the crossbar
            s = min_jerk((self.t - self.bounds[5]) / OPEN_STEPS)
            hand = HOLD_HAND + s * (OPEN_HAND - HOLD_HAND)
        else:
            hand = OPEN_HAND
        self.t += 1
        return {
            "action.right_arm": right_arm,
            "action.right_hand": hand,
            "action.left_arm": self.hold["left_arm"],
            "action.left_hand": self.hold["left_hand"],
            "action.waist": self.hold["waist"],
        }
