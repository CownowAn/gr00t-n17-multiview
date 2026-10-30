"""Scripted policy for ColorButtonBox: point an index finger and press the target.

The right hand presses, or the left one when the task turns the button face to the left
(ColorButtonBoxV3, ``env.face_side``); the other arm holds its start pose.

Uses privileged state (button pose, face normal), so the same trajectory is produced
whatever cameras are rendered. Phases, in tip space:
    approach: current tip -> pre-press point in front of the hood edge, along a curve
              through a via point
    settle, press straight along -normal (under the hood) past the face, hold,
    retract back to the pre-press point, then idle.
The waist stays fixed so that the egoview camera does not move.

reset(obs, rng) draws the trajectory parameters from TRAJECTORY_RANGES with ``rng``
(pass one seeded per episode for reproducible, varied demonstrations); without an
rng every parameter takes the middle of its range.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.tasks.color_button import BOX_PREFIX

# Hand action (either hand): [pinky, ring, middle, index, thumb bend, thumb rotation] (rad).
POINTING_HAND = np.array([1.5, 1.5, 1.5, 0.0, 0.6, 1.0])

# Low enough that the finger may tilt when pointing straight at the button is out of
# reach; the press itself still moves along -normal.
ORIENTATION_COST = 0.1
# Uniform ranges of the per-episode trajectory parameters. Durations are control steps
# (20 Hz); "press" lasts as long as the press distance takes at press_speed.
TRAJECTORY_RANGES = dict(
    approach=(60, 100),
    settle=(5, 15),
    hold=(10, 30),
    retract=(20, 35),
    hand_close=(12, 28),  # steps to form the pointing hand
    pre_clearance=(0.04, 0.07),  # pre-press point in front of the hood edge (m)
    # Past the resting face; the arm tracks ~8 mm behind the command (m).
    press_depth=(0.018, 0.025),
    press_speed=(0.002, 0.0035),  # tip travel per step while pressing (m)
    via_lateral=(-0.03, 0.03),  # via point offset from the approach midpoint (m)
    via_up=(0.0, 0.04),
)
DURATIONS = ("approach", "settle", "press", "hold", "retract")


def trajectory_rng(seed: int) -> np.random.Generator:
    """Trajectory rng of an episode, independent of the env rng that shares its seed."""
    return np.random.default_rng([seed, 1])


def sample_trajectory_params(rng=None) -> dict:
    params = {}
    for name, (lo, hi) in TRAJECTORY_RANGES.items():
        size = 2 if name == "via_lateral" else None  # world x and y
        value = np.full(size or (), (lo + hi) / 2) if rng is None else rng.uniform(lo, hi, size=size)
        params[name] = int(round(float(value))) if isinstance(lo, int) else value
    return params


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ColorButtonScriptedPolicy:
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    # ------------------------------------------------------------------
    def reset(self, obs: dict, rng=None):
        env = self.gym_env.env
        self.params = p = sample_trajectory_params(rng)
        p.update(getattr(env, "TRAJECTORY_FIXED", {}))  # task variants may fix some parameters
        model, data = env.sim.model._model, env.sim.data._data
        self.side = getattr(env, "face_side", "right")
        self.other = "left" if self.side == "right" else "right"
        link = f"gripper0_{self.side}_{self.side[0].upper()}_index_intermediate_link"
        self.ik = ArmIK(model, self.side, link, orientation_cost=ORIENTATION_COST)
        self.ik.reset(data.qpos)

        # Fingertip in the index intermediate link frame (see ColorButtonBox.index_tip).
        self.tip_local = 2.0 * model.geom_pos[model.geom(f"{link}_vis").id].copy()
        axis_local = self.tip_local / np.linalg.norm(self.tip_local)

        face = env.target_face_center()
        self.normal = data.xmat[model.body(f"{BOX_PREFIX}body").id].reshape(3, 3)[:, 0].copy()
        # The hood edge is `hood` past the box face; a face that sticks out past the box face
        # (FACE_PROTRUSION, e.g. an open drawer front) is that much nearer to it.
        pre_dist = env.variant_params["hood"] - getattr(env, "FACE_PROTRUSION", 0.0) + p["pre_clearance"]
        self.pre_point = face + pre_dist * self.normal
        self.press_point = face - p["press_depth"] * self.normal
        press_steps = int(np.ceil((pre_dist + p["press_depth"]) / p["press_speed"]))
        self.steps = {name: (press_steps if name == "press" else p[name]) for name in DURATIONS}
        self.bounds = np.cumsum([0] + list(self.steps.values()))

        # Orientation: rotate the current link frame so the finger points along -normal.
        start = self.ik.frame_pose()
        r_start = Rotation.from_matrix(start.rotation().as_matrix())
        align, _ = Rotation.align_vectors([-self.normal], [r_start.apply(axis_local)])
        self.r_goal = align * r_start
        self.slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_start, self.r_goal]))
        self.tip_start = start.translation() + r_start.apply(self.tip_local)
        # Via point of the quadratic Bezier approach: the midpoint, shifted sideways
        # (world x, y) and up.
        offset = np.array([*p["via_lateral"], p["via_up"]])
        self.via_point = (self.tip_start + self.pre_point) / 2 + offset

        self.hold = {
            k: np.asarray(obs[f"state.{k}"], dtype=np.float64).copy()
            for k in (f"{self.other}_arm", f"{self.other}_hand", "waist")
        }
        self.hand_start = np.asarray(obs[f"state.{self.side}_hand"], dtype=np.float64).copy()
        self.t = 0

    # ------------------------------------------------------------------
    def _target(self, t: int):
        """Tip position and link rotation for control step ``t``."""
        _, settle, press, hold, retract, _ = self.bounds
        if t < settle:
            s = min_jerk(t / self.steps["approach"])
            a, c, b = self.tip_start, self.via_point, self.pre_point
            return (1 - s) ** 2 * a + 2 * (1 - s) * s * c + s**2 * b, self.slerp(s)
        if t < press:
            return self.pre_point, self.r_goal
        if t < hold:
            s = (t - press) / self.steps["press"]
            return self.pre_point + s * (self.press_point - self.pre_point), self.r_goal
        if t < retract:
            return self.press_point, self.r_goal
        s = min_jerk((t - retract) / self.steps["retract"])
        return self.press_point + s * (self.pre_point - self.press_point), self.r_goal

    @property
    def total_steps(self) -> int:
        return int(self.bounds[-1])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        for name, end in zip(self.steps, self.bounds[1:]):
            if t < end:
                return name
        return "idle"

    def act(self) -> dict:
        tip, rot = self._target(min(self.t, self.total_steps))
        link_pos = tip - rot.apply(self.tip_local)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rot.as_matrix()), link_pos
        )
        arm = self.ik.solve(target)
        s = min_jerk(self.t / self.params["hand_close"])
        hand = self.hand_start + s * (POINTING_HAND - self.hand_start)
        self.t += 1
        return {
            f"action.{self.side}_arm": arm,
            f"action.{self.side}_hand": hand,
            f"action.{self.other}_arm": self.hold[f"{self.other}_arm"],
            f"action.{self.other}_hand": self.hold[f"{self.other}_hand"],
            "action.waist": self.hold["waist"],
        }

    def tip_error(self) -> float:
        """Distance between the commanded and the simulated fingertip (m)."""
        tip, _ = self._target(min(self.t - 1, self.total_steps))
        return float(np.linalg.norm(self.gym_env.env.index_tip(self.side) - tip))
