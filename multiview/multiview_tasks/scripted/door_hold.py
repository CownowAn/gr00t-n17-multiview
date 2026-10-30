"""Scripted policy for HoldDoorTakeBlock: reach past the held door into the cabinet,
take the named block and put it down on the work surface to the right.

Uses privileged state (block poses). The left arm and hand keep the pose the episode
starts in (holding the door). The right hand is steered by its grasp point (the env's
RIGHT_GRASP_SITE, where the closed fingertips meet, ~10 cm ahead of the palm), with the
palm down and the fingers pointing forward-left (45 deg); steering the wrist instead
let small orientation errors swing the grasp point onto a neighboring block. The joint paths are
planned once at reset by IK continuation along straight Cartesian segments (small
steps, each seeded with the previous solution), from the rest arm pose through every
waypoint in order, so the whole reach stays on one arm configuration (interpolating
separately solved configurations in joint space swept the arm into the cabinet):
    approach  straight up, then to beside the cabinet's open right side above the
              block's height, turning the hand from its rest orientation on the way
    enter     sideways into the cabinet, over the block
    descend   down onto the block
    close     close the hand and wait until the env welds the block (or time out)
    lift, withdraw (out through the right side), carry (to the drop point, down to the
    work surface), release (open the hand), retract

reset(obs, rng) draws the trajectory parameters (phase durations, waypoint offsets, hand
orientation, via points) from TRAJECTORY_RANGES with ``rng``; use trajectory_rng(seed),
which is independent of the env rng, so a seed gives the same scene and the same
trajectory at every level. Without an rng every parameter takes the middle of its range.
The grasp pose itself is not randomized (blocks are 8 cm apart, the grasp radius 5 cm).
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.tasks.door_hold import BLOCK_HALF, HOLD_HAND, RIGHT_GRASP_SITE

RIGHT_SITE = RIGHT_GRASP_SITE
OPEN_HAND = np.zeros(6)
ORIENTATION_COST = 0.3
DESCEND_STOP = 0.02  # stop the descent this far above the grasp pose; closing reaches down
DROP_FRU = np.array([0.30, 0.38, BLOCK_HALF + 0.012])  # block center when let go (range middle)
IK_ROUNDS = 60  # x5 IK iterations for the first pose
PATH_STEP = 0.01  # Cartesian spacing of the IK continuation (m)
PATH_ROUNDS = 6  # x5 IK iterations per continuation step

PHASE_NAMES = ("hold", "approach", "enter", "descend", "close", "lift", "withdraw", "carry", "release", "retract")
FIXED_DURATIONS = dict(hold=8, close=40)  # close ends early once the block is held
# Uniform ranges of the per-episode trajectory parameters; the middles are the values
# of the unrandomized policy. Durations are control steps (20 Hz); offsets are metres
# along the robot's (forward, right, up).
TRAJECTORY_RANGES = dict(
    approach=(36, 56),
    enter=(24, 38),
    descend=(20, 30),
    lift=(12, 19),
    withdraw=(24, 38),
    carry=(36, 56),
    release=(16, 24),
    retract=(20, 30),
    hand_close=(8, 16),  # steps to close the hand around the block
    raise_offset=((-0.04, 0.03, 0.13), (0.0, 0.07, 0.17)),  # first approach leg from the rest hand
    outside=((-0.045, 0.105, 0.0), (-0.015, 0.135, 0.0)),  # hand clear of the cabinet's right side, from above/lifted
    above=(0.05, 0.09),  # hand above its grasp pose while entering
    carry_up=(0.07, 0.11),  # lift before withdrawing, so the held block clears the others
    drop_offset=((-0.03, -0.03, 0.0), (0.03, 0.03, 0.0)),  # drop point offset from DROP_FRU
    finger_yaw=(35.0, 55.0),  # fingers this far left of forward (deg)
    carry_via=((-0.03, -0.03, 0.0), (0.03, 0.03, 0.04)),  # via point offset from the carry midpoint
)


def trajectory_rng(seed: int) -> np.random.Generator:
    """Trajectory rng of an episode, independent of the env rng that shares its seed."""
    return np.random.default_rng([seed, 1])


def sample_trajectory_params(rng=None) -> dict:
    params = {}
    for name, (lo, hi) in TRAJECTORY_RANGES.items():
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        value = (lo + hi) / 2 if rng is None else rng.uniform(lo, hi)
        if isinstance(TRAJECTORY_RANGES[name][0], int):
            value = int(round(float(value)))
        params[name] = value
    return params
SEGMENTS = {
    "hold": ("start", "start"),
    "approach": ("start", "front"),
    "enter": ("front", "above"),
    "descend": ("above", "grasp"),
    "close": ("grasp", "grasp"),
    "lift": ("grasp", "lifted"),
    "withdraw": ("lifted", "outside"),
    "carry": ("outside", "drop"),
    "release": ("drop", "drop"),
    "retract": ("drop", "retracted"),
}


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class HoldDoorScriptedPolicy:
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    def reset(self, obs: dict, rng=None):
        env = self.gym_env.env
        self.params = p = sample_trajectory_params(rng)
        self.phases = tuple((n, FIXED_DURATIONS.get(n) or p[n]) for n in PHASE_NAMES)
        model, data = env.sim.model._model, env.sim.data._data
        self.site = model.site(RIGHT_SITE).id
        base = data.xmat[model.body("robot0_base").id].reshape(3, 3)
        self.fwd, self.right, self.up = base[:, 0], -base[:, 1], np.array([0.0, 0.0, 1.0])
        self.fru = np.stack([self.fwd, self.right, self.up])
        # Rest hand: palm toward the robot's left, fingers up. The reach orientation is
        # that hand turned so the palm faces down and the fingers point forward-left.
        r0 = Rotation.from_matrix(data.site_xmat[self.site].reshape(3, 3))
        # Tasks may move some ranges (env.SCRIPTED_RANGES, same keys as TRAJECTORY_RANGES):
        # each drawn value is mapped linearly from its default range onto the task's, so the
        # rng stream (and every other parameter of a seed) stays the same.
        target_slot = env.block_colors[env.target_color]
        ranges = dict(getattr(env, "SCRIPTED_RANGES", {}))
        ranges.update(getattr(env, "SCRIPTED_SLOT_RANGES", {}).get(target_slot, {}))  # per target slot
        for name, (lo2, hi2) in ranges.items():
            lo, hi = (np.asarray(v, dtype=float) for v in TRAJECTORY_RANGES[name])
            frac = np.divide(p[name] - lo, hi - lo, out=np.full_like(lo, 0.5), where=hi != lo)
            p[name] = np.asarray(lo2, dtype=float) + frac * (np.asarray(hi2, dtype=float) - np.asarray(lo2, dtype=float))
        # Tasks may narrow the finger yaw for targets in the right slot column, where fingers
        # turned far left reach over the neighboring left-column slot (HoldDoorTakeBlockV2).
        right_col = getattr(env, "FINGER_YAW_RIGHT_COLUMN", None)
        slot_r = env.SLOTS[env.block_colors[env.target_color]][1]
        if right_col is not None and slot_r >= max(r for _, r in env.SLOTS):
            lo, hi = TRAJECTORY_RANGES["finger_yaw"]
            p["finger_yaw"] = right_col[0] + (p["finger_yaw"] - lo) / (hi - lo) * (right_col[1] - right_col[0])
        yaw = np.deg2rad(p["finger_yaw"])
        fingers = np.cos(yaw) * self.fwd - np.sin(yaw) * self.right
        self.r_hand = Rotation.align_vectors([-self.up, fingers], [-self.right, self.up])[0] * r0
        self.r_rest = r0
        self.target_color = env.target_color
        self.waypoints = {"start": data.site_xpos[self.site].copy()}
        self._plan()
        self._plan_paths(model, data.qpos.copy())
        self.bounds = np.cumsum([0] + [d for _, d in self.phases])
        self.hold = {
            "left_arm": env.left_arm_hold.copy(),
            "left_hand": HOLD_HAND.astype(np.float64),
            "waist": np.asarray(obs["state.waist"], dtype=np.float64).copy(),
        }
        self.t = 0

    def _plan_paths(self, model, qpos):
        """Joint-space path (n x 7) for every phase."""
        ik = ArmIK(model, "right", RIGHT_SITE, frame_type="site", orientation_cost=ORIENTATION_COST)
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
            """IK continuation along the polyline through waypoints ``names``."""
            pts = [w[n] for n in names]
            lengths = [np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])]
            total = max(sum(lengths), 1e-6)
            path, q, err, done = [q_seed], q_seed, 0.0, 0.0
            for a, b, length in zip(pts, pts[1:], lengths):
                n = max(1, int(np.ceil(length / PATH_STEP)))
                for s in np.linspace(0.0, 1.0, n + 1)[1:]:
                    frac = (done + s * length) / total
                    rot = turn(frac) if turn is not None else self.r_hand
                    q, e = solve(q, a + s * (b - a), rot, PATH_ROUNDS)
                    path.append(q)
                    err = max(err, e)
                done += length
            self.ik_error[phase] = err
            return np.array(path)

        q_start = qpos[idx].copy()
        turn = Slerp([0.0, 1.0], Rotation.concatenate([self.r_rest, self.r_hand]))
        self.paths = {"hold": np.array([q_start, q_start])}
        self.paths["approach"] = line(q_start, ["start", "raised", "front"], "approach", turn)
        self.paths["enter"] = line(self.paths["approach"][-1], ["front", "above"], "enter")
        self.paths["descend"] = line(self.paths["enter"][-1], ["above", "grasp"], "descend")
        q_grasp = self.paths["descend"][-1]
        self.paths["close"] = np.array([q_grasp, q_grasp])
        self.paths["lift"] = line(q_grasp, ["grasp", "lifted"], "lift")
        self.paths["withdraw"] = line(self.paths["lift"][-1], ["lifted", "outside"], "withdraw")
        self.paths["carry"] = line(self.paths["withdraw"][-1], ["outside", "carry_via", "drop"], "carry")
        q_drop = self.paths["carry"][-1]
        self.paths["release"] = np.array([q_drop, q_drop])
        self.paths["retract"] = line(q_drop, ["drop", "retracted"], "retract")


    def _plan(self):
        env = self.gym_env.env
        block = env.block_pos(self.target_color)
        w = self.waypoints
        p = self.params
        w["raised"] = w["start"] + p["raise_offset"] @ self.fru
        w["grasp"] = block + DESCEND_STOP * self.up
        w["above"] = w["grasp"] + p["above"] * self.up
        w["front"] = w["above"] + p["outside"] @ self.fru
        w["lifted"] = w["grasp"] + p["carry_up"] * self.up
        w["outside"] = w["lifted"] + p["outside"] @ self.fru
        # Drop height: the held item's center above the surface (DROP_FRU is for a block).
        drop = DROP_FRU + np.array([0.0, 0.0, env.item_half_height(self.target_color) - BLOCK_HALF])
        w["drop"] = env._world(*(drop + p["drop_offset"]))
        w["carry_via"] = (w["outside"] + w["drop"]) / 2 + p["carry_via"] @ self.fru
        w["retracted"] = w["drop"] + 0.10 * self.up + 0.05 * self.right

    @property
    def total_steps(self) -> int:
        return int(self.bounds[-1])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        for (name, _), end in zip(self.phases, self.bounds[1:]):
            if t < end:
                return name
        return "idle"

    def _target(self, t: int, table=None):
        table = self.waypoints if table is None else table
        name = self.phase(t)
        if name == "idle":
            return table["retracted"], name
        i = PHASE_NAMES.index(name)
        s = (t - self.bounds[i]) / self.phases[i][1]
        a, b = (table[k] for k in SEGMENTS[name])
        return a + min_jerk(s) * (b - a), name

    def act(self) -> dict:
        env = self.gym_env.env
        name = self.phase()
        if name == "close" and env._grasped is not None and self.t < self.bounds[5] - 5:
            self.t = self.bounds[5] - 5  # held: move on after a few settling steps
        if name == "idle":
            right_arm = self.paths["retract"][-1]
        else:
            i = PHASE_NAMES.index(name)
            path = self.paths[name]
            u = min_jerk((self.t - self.bounds[i]) / self.phases[i][1]) * (len(path) - 1)
            k = min(int(u), len(path) - 2)
            right_arm = path[k] + (u - k) * (path[k + 1] - path[k])
        if name == "close":
            s = min_jerk((self.t - self.bounds[4]) / self.params["hand_close"])
            hand = s * HOLD_HAND
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

    def tip_error(self) -> float:
        env = self.gym_env.env
        return float(np.linalg.norm(env.block_pos(self.target_color) - env.grasp_point()))
