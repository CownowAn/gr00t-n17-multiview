"""Scripted policy for ShelfTakeBlock: reach into the low shelf through its opening, take
the named block and put it down on the work surface to the right of the shelf.

Uses privileged state (block poses). The right hand is steered by its grasp point (the
env's RIGHT_GRASP_SITE, where the closed fingertips meet), palm down, fingers pointing
forward and somewhat left. Joint paths are planned once at reset by IK continuation
along Cartesian polylines from the rest arm pose (as in scripted/door_hold.py):
    approach  back toward the body first (the resting hand is up beside the shelf's
              front corner, so a direct path cuts through it), then via a random point
              down in front of the opening, turning the hand from its rest orientation
              on the way; in some episodes it ends a few cm beside the target slot
    correct   sideways onto the slot (zero length unless the approach was offset)
    enter     straight in, over the block
    descend   down onto the block
    close     close the hand and wait until the env welds the block (or time out)
    lift (a little: the shelf is 14 cm high), withdraw (straight out), carry (via a
    random point to the drop point right of the shelf, down to the work surface),
    release, retract

reset(obs, rng) draws the trajectory parameters from TRAJECTORY_RANGES with ``rng``; use
trajectory_rng(seed) (independent of the env rng). Without an rng every parameter takes
the middle of its range and there is no offset approach.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.tasks.shelf_take import BLOCK_HALF, RIGHT_GRASP_SITE, ROOF_OVERHANG

OPEN_HAND = np.zeros(6)
HOLD_HAND = np.array([1.0, 1.0, 1.0, 1.0, 0.7, 1.0])
ORIENTATION_COST = 0.05  # the grasp point is the IK frame, so orientation slack does not move it
DESCEND_STOP = 0.02  # stop the descent this far above the block; closing reaches down
DROP_FRU = np.array([0.22, 0.36, BLOCK_HALF + 0.012])  # block center when let go (range middle)
OFFSET_APPROACH_PROB = 0.25  # episodes that arrive beside the slot and correct sideways
IK_ROUNDS = 60
PATH_STEP = 0.01
PATH_ROUNDS = 6

PHASE_NAMES = ("hold", "approach", "correct", "enter", "descend", "close", "lift", "withdraw", "carry", "release",
               "retract")
FIXED_DURATIONS = dict(hold=8, close=40)
TRAJECTORY_RANGES = dict(
    approach=(44, 72),
    correct=(10, 18),  # only used by offset approaches
    enter=(22, 40),
    descend=(14, 24),
    lift=(10, 16),
    withdraw=(22, 40),
    carry=(30, 52),
    release=(14, 24),
    retract=(18, 30),
    hand_close=(8, 16),
    retreat_fwd=(0.14, 0.20),  # grasp point's forward coordinate after leaving the rest pose
    retreat_offset=((0.0, -0.03, 0.0), (0.0, 0.03, 0.05)),  # (unused, right, up) from the rest hand
    via=((-0.03, -0.05, -0.02), (0.03, 0.05, 0.05)),  # bend of the descent toward the opening
    pre_gap=(0.04, 0.09),  # grasp point this far in front of the shelf top's front edge before entering
    offset=(0.025, 0.045),  # sideways miss of an offset approach (random side)
    enter_height=(0.045, 0.06),  # grasp point above the block while entering and withdrawing
    lift_height=(0.02, 0.03),  # lift before withdrawing (the shelf is 14 cm high)
    drop_offset=((-0.04, -0.04, 0.0), (0.04, 0.04, 0.0)),
    finger_yaw=(15.0, 35.0),  # fingers this far left of forward (deg)
    carry_via=((-0.03, -0.03, 0.02), (0.02, 0.03, 0.08)),
    ease=(0.8, 1.25),  # per-phase time warp exponent (slower start or slower end)
)

SEGMENTS = {
    "hold": ("start",),
    "approach": ("start", "retreat", "via", "pre_offset"),
    "correct": ("pre_offset", "pre"),
    "enter": ("pre", "above"),
    "descend": ("above", "grasp"),
    "close": ("grasp",),
    "lift": ("grasp", "lifted"),
    "withdraw": ("lifted", "outside"),
    "carry": ("outside", "carry_via", "drop"),
    "release": ("drop",),
    "retract": ("drop", "retracted"),
}


def trajectory_rng(seed: int) -> np.random.Generator:
    """Trajectory rng of an episode, independent of the env rng that shares its seed."""
    return np.random.default_rng([seed, 1])


def sample_trajectory_params(rng=None) -> dict:
    params = {}
    for name, (lo, hi) in TRAJECTORY_RANGES.items():
        lo_a, hi_a = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        value = (lo_a + hi_a) / 2 if rng is None else rng.uniform(lo_a, hi_a)
        if isinstance(lo, int):
            value = int(round(float(value)))
        params[name] = value
    if rng is None:
        params["offset_side"] = 0.0
        params["ease_per_phase"] = {n: 1.0 for n in PHASE_NAMES}
    else:
        params["offset_side"] = float(rng.choice([-1.0, 1.0])) if rng.uniform() < OFFSET_APPROACH_PROB else 0.0
        lo, hi = TRAJECTORY_RANGES["ease"]
        params["ease_per_phase"] = {n: float(rng.uniform(lo, hi)) for n in PHASE_NAMES}
    return params


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ShelfTakeScriptedPolicy:
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    def reset(self, obs: dict, rng=None):
        env = self.gym_env.env
        self.params = p = sample_trajectory_params(rng)
        durations = {n: FIXED_DURATIONS.get(n) or p[n] for n in PHASE_NAMES}
        if not p["offset_side"]:
            durations["correct"] = 0
        self.phases = tuple((n, durations[n]) for n in PHASE_NAMES)
        model, data = env.sim.model._model, env.sim.data._data
        self.site = model.site(RIGHT_GRASP_SITE).id
        base = data.xmat[model.body("robot0_base").id].reshape(3, 3)
        self.fwd, self.right, self.up = base[:, 0], -base[:, 1], np.array([0.0, 0.0, 1.0])
        self.fru = np.stack([self.fwd, self.right, self.up])
        r0 = Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3))
        yaw = np.deg2rad(p["finger_yaw"])
        fingers = np.cos(yaw) * self.fwd - np.sin(yaw) * self.right
        self.r_hand = Rotation.align_vectors([-self.up, fingers], [-self.right, self.up])[0] * r0
        self.r_rest = r0
        self.target_color = env.target_color
        self.waypoints = {"start": data.site_xpos[self.site].copy()}
        self._plan()
        self._plan_paths(model, data.qpos.copy())
        self.bounds = np.cumsum([0] + [d for _, d in self.phases])
        self.hold = {k: np.asarray(obs[f"state.{k}"], dtype=np.float64).copy() for k in ("left_arm", "left_hand", "waist")}
        self.t = 0

    def _plan(self):
        env = self.gym_env.env
        p, w = self.params, self.waypoints
        block = env.block_pos(self.target_color)
        f, r, _ = env._robot_frame(block)
        f0, r0, z0 = env._robot_frame(w["start"])
        # The top's front edge (also where the side walls would catch a forearm coming down).
        edge = env.mouth - ROOF_OVERHANG
        w["grasp"] = block + DESCEND_STOP * self.up
        w["above"] = block + p["enter_height"] * self.up
        w["pre"] = w["above"] - (f - edge + p["pre_gap"]) * self.fwd
        w["pre_offset"] = w["pre"] + p["offset_side"] * p["offset"] * self.right
        w["retreat"] = env._world(p["retreat_fwd"], r0 + p["retreat_offset"][1], z0 + p["retreat_offset"][2])
        w["via"] = (w["retreat"] + w["pre_offset"]) / 2 + p["via"] @ self.fru
        w["lifted"] = w["grasp"] + p["lift_height"] * self.up
        w["outside"] = w["lifted"] - (f - edge + p["pre_gap"]) * self.fwd
        w["drop"] = env._world(*(DROP_FRU + p["drop_offset"]))
        w["carry_via"] = (w["outside"] + w["drop"]) / 2 + p["carry_via"] @ self.fru
        w["retracted"] = w["drop"] + 0.10 * self.up - 0.05 * self.fwd

    def _plan_paths(self, model, qpos):
        ik = ArmIK(model, "right", RIGHT_GRASP_SITE, frame_type="site", orientation_cost=ORIENTATION_COST)
        idx = ik.qpos_idx
        w = self.waypoints
        self.ik_error = {}

        def solve(q_seed, pos, rot, rounds):
            q = qpos.copy()
            q[idx] = q_seed
            ik.reset(q)
            goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot.as_matrix()), pos)
            for _ in range(rounds):
                sol = ik.solve(goal)
            return sol, float(np.linalg.norm(ik.frame_pose().translation() - pos))

        def line(q_seed, names, phase, turn=None):
            pts = [w[n] for n in names]
            lengths = [np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])]
            total = max(sum(lengths), 1e-6)
            path, q, err, done = [q_seed], q_seed, 0.0, 0.0
            for a, b, length in zip(pts, pts[1:], lengths):
                n = max(1, int(np.ceil(length / PATH_STEP)))
                for s in np.linspace(0.0, 1.0, n + 1)[1:]:
                    rot = turn((done + s * length) / total) if turn is not None else self.r_hand
                    q, e = solve(q, a + s * (b - a), rot, PATH_ROUNDS)
                    path.append(q)
                    err = max(err, e)
                done += length
            self.ik_error[phase] = err
            return np.array(path)

        turn = Slerp([0.0, 1.0], Rotation.concatenate([self.r_rest, self.r_hand]))
        q = qpos[idx].copy()
        self.paths = {}
        for name in PHASE_NAMES:
            names = SEGMENTS[name]
            if len(names) == 1:
                self.paths[name] = np.array([q, q])
            else:
                self.paths[name] = line(q, names, name, turn if name == "approach" else None)
            q = self.paths[name][-1]

    @property
    def total_steps(self) -> int:
        return int(self.bounds[-1])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        for (name, _), end in zip(self.phases, self.bounds[1:]):
            if t < end:
                return name
        return "idle"

    def act(self) -> dict:
        env = self.gym_env.env
        name = self.phase()
        i_close = PHASE_NAMES.index("close")
        if name == "close" and env._grasped is not None and self.t < self.bounds[i_close + 1] - 5:
            self.t = self.bounds[i_close + 1] - 5  # held: move on after a few settling steps
        if name == "idle":
            right_arm = self.paths["retract"][-1]
        else:
            i = PHASE_NAMES.index(name)
            path = self.paths[name]
            s = (self.t - self.bounds[i]) / self.phases[i][1]
            u = min_jerk(s ** self.params["ease_per_phase"][name]) * (len(path) - 1)
            k = min(int(u), len(path) - 2)
            right_arm = path[k] + (u - k) * (path[k + 1] - path[k])
        if name == "close":
            hand = min_jerk((self.t - self.bounds[i_close]) / self.params["hand_close"]) * HOLD_HAND
        elif name in ("lift", "withdraw", "carry"):
            hand = HOLD_HAND
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
