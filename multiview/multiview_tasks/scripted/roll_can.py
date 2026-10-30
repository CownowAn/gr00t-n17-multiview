"""Scripted policy for StopRollingCan: see the can start rolling, then put a hand in its path.

Uses privileged state (which can rolls, along which line, and when it is pushed), so the
same trajectory is produced whatever cameras are rendered. Both hands wait at rest.
REACTION steps after the push, the hand on the can's side rises, moves over at that
height to above a blocking pose on the can's path and then comes down onto the counter, between the can and the midline: the hand stands on
its little-finger edge with the fingers pointing forward along the can, the back of the
hand toward the oncoming can, low enough that the can cannot pass under it. The hand
stays there until the can has stopped, then holds.

reset(obs, rng) draws the trajectory parameters from TRAJECTORY_RANGES with ``rng``
(pass one seeded per episode for reproducible, varied demonstrations); without an rng
every parameter takes the middle of its range.
"""

import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.tasks.catch_ball import PALMS, SIDES

BLOCK_HAND = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 1.0])  # fingers straight, thumb tucked
TRAJECTORY_RANGES = dict(
    reaction=(4, 8),  # steps from the push to the first motion
    approach=(18, 24),  # steps to the pose above the blocking pose
    lower=(6, 10),  # steps down onto the counter
    lift=(0.08, 0.10),  # the hand comes in this high above the blocking pose (clear of the counter and the gutter)
    block_side=(0.20, 0.28),  # |right| of the blocking hand's palm (m)
    block_height=(0.045, 0.055),  # palm origin above the work surface (m)
    block_back=(0.06, 0.08),  # palm origin this far behind the can's line (fingers reach past it)
)
MIRROR = np.array([1, -1, -1, 1, -1, -1, 1])  # left <-> right arm joint signs
NOMINAL_ARM = np.array([-0.3, 0.2, 0.0, -1.6, 0.0, 0.0, 0.0])  # left arm reaching forward and down (an extra IK seed)


def trajectory_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 1])


def sample_trajectory_params(rng=None, ranges=TRAJECTORY_RANGES) -> dict:
    params = {}
    for name, (lo, hi) in ranges.items():
        value = (lo + hi) / 2 if rng is None else rng.uniform(lo, hi)
        params[name] = int(round(value)) if isinstance(lo, int) else float(value)
    return params


def min_jerk(s: float) -> float:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class RollCanScriptedPolicy:
    RANGES = TRAJECTORY_RANGES
    def __init__(self, gym_env):
        """``gym_env``: the unwrapped MultiViewGrootEnv (dict actions)."""
        self.gym_env = gym_env

    def reset(self, obs: dict, rng=None, side: str = None, start_step: int = None):
        """side / start_step override the hand that blocks and the step the motion starts
        (for perception-limited oracles); by default they come from the layout."""
        env = self.gym_env.env
        self.params = p = sample_trajectory_params(rng, self.RANGES)
        self.side = side or env.rolling_side
        sgn = SIDES[self.side]
        self.model = env.sim.model._model
        data = env.sim.data._data
        self.qpos0 = data.qpos.copy()
        target, self.fingers = self._geometry(env, p)
        self.iks = {s: ArmIK(self.model, s, PALMS[s], frame_type="body", orientation_cost=0.3) for s in SIDES}
        self.q_rest = self.qpos0[self.iks[self.side].qpos_idx].copy()
        self.approach_path, self.q_block, self.ik_error = self._plan_approach(env, target, p["lift"])
        self.start = (env.layout["push_step"] + p["reaction"]) if start_step is None else start_step
        self.rest = {k: np.asarray(obs[f"state.{k}"], dtype=np.float64).copy()
                     for k in ("left_arm", "right_arm", "left_hand", "right_hand", "waist")}
        self.t = 0

    def _geometry(self, env, p):
        """(palm target (world), {side: finger direction (world, horizontal)}): the can rolls
        sideways along its lane, so the fingers point forward along it, the palm origin
        block_back behind the lane."""
        sgn = SIDES[self.side]
        _, _, fwd, _, _ = env.frame
        lane = env.layout[f"lane_{self.side}"]
        return env._world(lane - p["block_back"], sgn * p["block_side"], p["block_height"]), {s: fwd for s in SIDES}

    def _block_rotation(self, env, side):
        """Palm orientation: the resting hand (palm toward the midline, fingers up) turned so
        the fingers point along self.fingers[side], the palm still facing as far toward the
        midline as it can (the hand stands on its little-finger edge)."""
        _, _, _, rgt, _ = env.frame
        up = np.array([0.0, 0.0, 1.0])
        inward = -SIDES[side] * rgt
        f = self.fingers[side]
        n = np.cross(up, f)
        n = n if n @ inward >= 0 else -n
        r_now = Rotation.from_matrix(env.sim.data._data.xmat[self.model.body(PALMS[side]).id].reshape(3, 3))
        turn = Rotation.align_vectors([n, f], [inward, up])[0]
        return (turn * r_now).as_matrix()

    def _solve(self, env, target):
        """Blocking joints (and palm position error, m), the most accurate of several IK
        seeds: the rest pose, a nominal forward reach, a solution 10 cm higher, and the other
        arm's mirrored solution (any one can stall; the right arm's rest seed can also land
        elbow-out, so it prefers an accurate mirrored one)."""
        run = lambda side, target, seed: self._run(env, side, target, seed)

        own = run(self.side, target, None)
        nominal = run(self.side, target, NOMINAL_ARM * (MIRROR if self.side == "right" else 1))
        above = run(self.side, target + np.array([0.0, 0.0, 0.10]), None)
        continued = run(self.side, target, above[0])
        _, _, _, rgt, _ = env.frame
        base = env._world(0.0, 0.0, 0.0)
        other = "left" if self.side == "right" else "right"
        q_other, _ = run(other, target - 2 * ((target - base) @ rgt) * rgt, None)
        mirrored = run(self.side, target, MIRROR * q_other)
        if self.side == "right" and mirrored[1] < 0.01:
            return mirrored  # the right arm's own solution can be elbow-out
        cands = (own, nominal, continued, mirrored)
        good = [c for c in cands if c[1] < 0.01] or [min(cands, key=lambda c: c[1])]
        # nearest the rest pose: a far-off branch swings the hand through the counter on the way
        return min(good, key=lambda c: float(np.linalg.norm(c[0] - self.q_rest)))

    def _plan_approach(self, env, target, lift):
        """Joint path from the start pose to the pose `lift` above the blocking pose, by IK
        continuation from the start pose (so it stays on the start pose's branch): the palm
        rises, moves over at that height and the hand turns gradually into the blocking
        orientation. Returns (path, blocking joints, blocking position error)."""
        ik = self.iks[self.side]
        q = self.qpos0.copy()
        q[ik.qpos_idx] = self.q_rest
        ik.reset(q)
        pose0 = ik.frame_pose()
        palm0 = pose0.translation()
        r0 = Rotation.from_matrix(pose0.rotation().as_matrix())
        r1 = Rotation.from_matrix(self._block_rotation(env, self.side))
        turn = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        above = target + np.array([0.0, 0.0, lift])
        high = np.array([palm0[0], palm0[1], max(above[2], palm0[2] + 0.04)])
        pts = [palm0 + (high - palm0) * k / 3 for k in range(1, 4)] + [high + (above - high) * k / 10 for k in range(1, 11)]
        path, q = [self.q_rest], self.q_rest
        for k, pt in enumerate(pts):
            rot = turn(min(1.0, (k + 1) / (0.7 * len(pts)))).as_matrix()
            q = self._run(env, self.side, pt, q, iters=60, rot=rot)[0]
            path.append(q)
        q_block, err = self._run(env, self.side, target, q)
        return np.array(path), q_block, err

    @staticmethod
    def _along(path, u):
        n = len(path) - 1
        x = float(np.clip(u, 0.0, 1.0)) * n
        k = min(int(x), n - 1)
        return path[k] + (x - k) * (path[k + 1] - path[k])

    def _run(self, env, side, target, seed, iters=200, rot=None):
        """IK for `side`'s palm at `target` in the blocking orientation from joint seed `seed`
        (None: the start pose); returns (joints, position error m)."""
        ik = self.iks[side]
        rot = self._block_rotation(env, side) if rot is None else rot
        goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), target)
        q = self.qpos0.copy()
        if seed is not None:
            q[ik.qpos_idx] = seed
        ik.reset(q)
        for _ in range(iters):
            sol = ik.solve(goal)
        return sol.copy(), float(np.linalg.norm(ik.frame_pose().translation() - target))

    @property
    def total_steps(self) -> int:
        return int(self.start + self.params["approach"] + self.params["lower"])

    def phase(self, t: int = None) -> str:
        t = self.t if t is None else t
        if t < self.start:
            return "wait"
        if t < self.start + self.params["approach"]:
            return "approach"
        return "lower" if t < self.total_steps else "block"

    def act(self) -> dict:
        action = {f"action.{k}": v.copy() for k, v in self.rest.items()}
        if self.t >= self.start:
            n = self.params["approach"]
            s = min_jerk((self.t - self.start) / n)
            u = min_jerk((self.t - self.start - n) / self.params["lower"])
            q_above = self.approach_path[-1]
            arm = self._along(self.approach_path, s) if u <= 0 else q_above + u * (self.q_block - q_above)
            action[f"action.{self.side}_arm"] = arm
            hand0 = self.rest[f"{self.side}_hand"]
            action[f"action.{self.side}_hand"] = hand0 + min(1.0, 2 * s) * (BLOCK_HAND - hand0)
        self.t += 1
        return action


class RollCanDirsScriptedPolicy(RollCanScriptedPolicy):
    """For StopRollingCanDirs: the can may come from ahead, the side or behind, along a
    path past a block point B. The hand stands in the path, block_side (read as a distance
    along the path) past B, with the fingers across the path, pointing to whichever side
    of it is more forward (for the diagonal front path, straight forward). Cans arrive faster here, so the hand moves
    faster."""

    RANGES = {**TRAJECTORY_RANGES, "approach": (13, 16), "lower": (5, 7)}

    def _geometry(self, env, p):
        _, _, fwd, rgt, _ = env.frame
        b, d = env.path(env.target if self.side == env.rolling_side else f"{self.side}_side")
        u = p["block_side"] - 0.18  # 0.02-0.10 m past B
        across = np.array([-d[1], d[0]])
        across = across if across[0] >= 0 else -across  # the more forward of the two
        if env.direction == "front":
            across = np.array([1.0, 0.0])  # turned across this diagonal path the hand is out of reach; at 40 deg it still blocks
        point = b + u * d - p["block_back"] * across
        world = lambda v: v[0] * fwd + v[1] * rgt
        mirror = lambda v: np.array([v[0], -v[1]])
        fingers = {self.side: world(across), ("left" if self.side == "right" else "right"): world(mirror(across))}
        return env._world(point[0], point[1], p["block_height"]), fingers
