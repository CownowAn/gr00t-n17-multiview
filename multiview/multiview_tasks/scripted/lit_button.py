"""Scripted policy for PressLitButton: wait for the light, then press it with that side's index finger.

Uses privileged state (which button lights up and when, its pose), so the same trajectory
is produced whatever cameras are rendered. Both hands wait at rest. REACTION steps after
the light comes on, the hand on the button's side forms a pointing hand and moves (in
joint space) to a stand-off pose STANDOFF in front of the button, then straight in along
-normal to a pre-press point, presses, holds, and backs out the same way. Coming in along
the normal keeps the hand off the panel and the other buttons. The other arm, the other
hand and the waist keep their start pose.

The joint path is planned at reset: the pre-press pose is solved from the rest pose and
from the mirrored solution of the other arm (stepwise IK on a moving target can drift
onto an elbow-out branch of the right arm), keeping the accurate one nearest the rest
pose; the straight lines in and out follow by IK continuation from it.

reset(obs, rng) draws the trajectory parameters from TRAJECTORY_RANGES with ``rng``
(pass one seeded per episode for reproducible, varied demonstrations); without an rng
every parameter takes the middle of its range.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from multiview_tasks.scripted.ik import ArmIK

# Hand action: [pinky, ring, middle, index, thumb bend, thumb rotation] (rad).
POINTING_HAND = np.array([1.5, 1.5, 1.5, 0.0, 0.6, 1.0])
ORIENTATION_COST = 0.1  # the finger may tilt where pointing straight in is out of reach
TRAJECTORY_RANGES = dict(
    reaction=(4, 8),  # steps from the light coming on to the first motion
    approach=(26, 40),
    settle=(8, 14),  # let the arm catch up with the pre-press pose (it lags under load)
    hold=(6, 12),
    retract=(16, 26),
    pre_clearance=(0.04, 0.06),  # pre-press point in front of the button face (m)
    press_depth=(0.024, 0.030),  # past the resting face; the extended arm lags 1-1.5 cm under load
    press_speed=(0.003, 0.004),  # tip travel per step while pressing (m)
    standoff=(0.18, 0.22),  # stand-off point in front of the button face (m); nearer, the hand brushed other buttons
    standoff_shift=(-0.03, 0.03),  # sideways and vertical shift of the stand-off point (m)
)
DURATIONS = ("approach", "enter", "settle", "press", "hold", "retract")
ENTER_STEPS = 14  # stand-off -> pre-press
MIRROR = np.array([1, -1, -1, 1, -1, -1, 1])  # left <-> right arm joint signs
PRESS_POINTS = 8  # IK continuation points along the press line


def trajectory_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 1])


def sample_trajectory_params(rng=None) -> dict:
    params = {}
    for name, (lo, hi) in TRAJECTORY_RANGES.items():
        size = 2 if name == "standoff_shift" else None
        value = np.full(size or (), (lo + hi) / 2) if rng is None else rng.uniform(lo, hi, size=size)
        params[name] = int(round(float(value))) if isinstance(lo, int) else value
    return params


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def index_link(side: str) -> str:
    return f"gripper0_{side}_{side[0].upper()}_index_intermediate_link"


class LitButtonScriptedPolicy:
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    def reset(self, obs: dict, rng=None, button: str = None, start_step: int = None):
        """button / start_step override the lit button and the step the motion starts
        (for perception-limited oracles); by default they come from the layout."""
        env = self.gym_env.env
        self.params = p = sample_trajectory_params(rng)
        self.button = button or env.target
        self.side = self.button.split("_")[0]
        self.model = env.sim.model._model
        self.qpos0 = env.sim.data._data.qpos.copy()
        self.iks = {s: ArmIK(self.model, s, index_link(s), orientation_cost=ORIENTATION_COST) for s in ("left", "right")}
        self.tip_local = {s: 2.0 * self.model.geom_pos[self.model.geom(f"{index_link(s)}_vis").id].copy()
                          for s in ("left", "right")}

        face = env.button_face_center(self.button)
        normal = env.button_normal(self.button)
        _, _, fwd, rgt, _ = env.frame
        side_dir = np.cross(np.array([0.0, 0.0, 1.0]), normal)
        standoff = face + p["standoff"] * normal + p["standoff_shift"][0] * side_dir + p["standoff_shift"][1] * np.array([0, 0, 1.0])
        pre_point = face + p["pre_clearance"] * normal
        press_point = face - p["press_depth"] * normal
        self.q_rest = self.qpos0[self.iks[self.side].qpos_idx].copy()
        self.q_pre, self.pre_error = self._solve_pre(env, pre_point, normal)
        self.q_enter = self._line(self.q_pre, pre_point, standoff, normal)[::-1]  # standoff -> pre
        self.q_press_path = self._line(self.q_pre, pre_point, press_point, normal)  # pre -> press

        press_steps = int(np.ceil((p["pre_clearance"] + p["press_depth"]) / p["press_speed"]))
        self.steps = {name: (press_steps if name == "press" else ENTER_STEPS if name == "enter" else p[name])
                      for name in DURATIONS}
        self.start = (env.layout["light_step"] + p["reaction"]) if start_step is None else start_step
        self.bounds = self.start + np.cumsum([0] + list(self.steps.values()))
        self.rest = {k: np.asarray(obs[f"state.{k}"], dtype=np.float64).copy()
                     for k in ("left_arm", "right_arm", "left_hand", "right_hand", "waist")}
        self.t = 0

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _goal(self, side, tip, normal):
        """Link pose putting `side`'s index fingertip at `tip` with the finger along -normal."""
        ik = self.iks[side]
        ik.reset(self.qpos0.copy())
        r_start = Rotation.from_matrix(ik.frame_pose().rotation().as_matrix())
        axis_local = self.tip_local[side] / np.linalg.norm(self.tip_local[side])
        align, _ = Rotation.align_vectors([-normal], [r_start.apply(axis_local)])
        rot = align * r_start
        return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot.as_matrix()),
                                                       tip - rot.apply(self.tip_local[side]))

    def _solve(self, side, tip, normal, q_seed, iters=150):
        """(joints, fingertip error m) from `q_seed`."""
        ik = self.iks[side]
        goal = self._goal(side, tip, normal)
        q = self.qpos0.copy()
        q[ik.qpos_idx] = q_seed
        ik.reset(q)
        for _ in range(iters):
            sol = ik.solve(goal)
        pose = ik.frame_pose()
        tip_now = pose.translation() + pose.rotation().as_matrix() @ self.tip_local[side]
        return sol.copy(), float(np.linalg.norm(tip_now - tip))

    def _line(self, q_from, a, b, normal, step=0.01):
        """IK continuation from `q_from` (fingertip at a) along the straight line a -> b."""
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        path, q = [q_from], q_from
        for s in np.linspace(0, 1, n + 1)[1:]:
            q, _ = self._solve(self.side, a + s * (b - a), normal, q, iters=40)
            path.append(q)
        return np.array(path)

    def _solve_pre(self, env, tip, normal):
        """Pre-press joints, from the rest pose and from the other arm's mirrored solution
        for the mirrored point; the most accurate (ties: nearest the rest pose)."""
        side = self.side
        other = "left" if side == "right" else "right"
        _, _, _, rgt, _ = env.frame
        base = env._world(0.0, 0.0, 0.0)
        mirror_tip = tip - 2 * ((tip - base) @ rgt) * rgt
        mirror_normal = normal - 2 * (normal @ rgt) * rgt
        q_other, _ = self._solve(other, mirror_tip, mirror_normal, self.qpos0[self.iks[other].qpos_idx].copy(), iters=300)
        mirrored = self._solve(side, tip, normal, MIRROR * q_other, iters=300)
        if side == "right" and mirrored[1] < 0.004:
            return mirrored  # the left arm's solutions press reliably; the right arm's own can be elbow-out
        cands = [self._solve(side, tip, normal, self.q_rest, iters=300), mirrored]
        good = [c for c in cands if c[1] < 0.004] or [min(cands, key=lambda c: c[1])]
        return min(good, key=lambda c: float(np.linalg.norm(c[0] - self.q_rest)))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    @staticmethod
    def _along(path, u):
        """Joint targets at fraction u in [0, 1] of a precomputed path."""
        n = len(path) - 1
        x = float(np.clip(u, 0.0, 1.0)) * n
        k = min(int(x), n - 1)
        return path[k] + (x - k) * (path[k + 1] - path[k])

    def _arm_target(self, t: int):
        """Joint targets of the moving arm at control step ``t`` (None before the motion)."""
        a, enter, settle, press, hold, retract, end = self.bounds
        if t < a:
            return None
        if t < enter:
            s = min_jerk((t - a) / self.steps["approach"])
            return self.q_rest + s * (self.q_enter[0] - self.q_rest)
        if t < settle:
            return self._along(self.q_enter, min_jerk((t - enter) / self.steps["enter"]))
        if t < press:
            return self.q_pre
        if t < hold:
            return self._along(self.q_press_path, (t - press) / self.steps["press"])
        if t < retract:
            return self.q_press_path[-1]
        # back out: press -> pre -> stand-off
        u = min_jerk((t - retract) / self.steps["retract"])
        out = np.concatenate([self.q_press_path[::-1], self.q_enter[::-1][1:]])
        return self._along(out, u)

    @property
    def total_steps(self) -> int:
        return int(self.bounds[-1])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        if t < self.bounds[0]:
            return "wait"
        for name, end in zip(self.steps, self.bounds[1:]):
            if t < end:
                return name
        return "idle"

    def act(self) -> dict:
        action = {f"action.{k}": v.copy() for k, v in self.rest.items()}
        q = self._arm_target(min(self.t, self.total_steps))
        if q is not None:
            action[f"action.{self.side}_arm"] = q
            s = min_jerk((self.t - self.bounds[0]) / max(1, self.steps["approach"] // 2))
            hand0 = self.rest[f"{self.side}_hand"]
            action[f"action.{self.side}_hand"] = hand0 + s * (POINTING_HAND - hand0)
        self.t += 1
        return action
