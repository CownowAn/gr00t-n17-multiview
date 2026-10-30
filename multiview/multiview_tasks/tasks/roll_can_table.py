"""Stop the can rolling toward you across one table (field of view, reaction).

One long table in front of the robot, running out to both sides, with one soda can on it.
The can waits where the head camera cannot see it, at the start of a straight path toward
a block point B ahead of one shoulder, coming either from the far side of the table
("front", nearly straight toward the robot), diagonally from a far corner ("diag") or along
the table from its end, slightly from behind the shoulder line ("side", half the
episodes); the side, the direction, and within each direction the path's angle (+-6 deg)
and length (0.70-0.80 m) are random. At a random moment the can is pushed and rolls along
its path at a fixed speed; unstopped, it rolls past B and on across the robot's midline.
The hand on that side has to get into its path and stop it: "stop the rolling can". The
hands wait at rest until the can rolls: a hand that moves earlier is a false start (fails).

What has to be perceived: where the can comes from and when it starts. The head camera
(in the bg-crop framing the policy reads) sees only the table near the hands, so it sees
the can only shortly before it arrives, too late to block it. Cameras on the body looking
diagonally out see a front or diagonal can wait and start; cameras looking straight out to
the sides see a side can. The hands are slow (about 0.5 s per 10 cm).

Knob: the can speed.

Success: the rolling can stays on the table, on its own side of the midline, and stops
moving along its path (it may rest against the hand or roll slowly back).
"""

import numpy as np
from robocasa.environments.tabletop.tabletop import Tabletop

from multiview_tasks.cameras import ROLL_CAMERAS
from multiview_tasks.tasks.catch_ball import PALMS, SIDES
from multiview_tasks.tasks.roll_can import CAN_RADIUS, COUNTER_RGBA, PLATFORM_HEIGHT, PUSH_STEPS, SETTLE_STEPS, SURFACE
from multiview_tasks.tasks.roll_can_dirs import StopRollingCanDirs
from multiview_tasks.variant import make_level_classes

PREFIX = "mvcand_"  # the can body and joint names of StopRollingCanDirs
# The table (robot frame: forward, right; metres from the robot base), one piece.
TABLE_FORWARD = (0.18, 1.25)
TABLE_HALF_WIDTH = 1.20
MIDLINE_MARGIN = 0.05  # the stopped can has to be at least this far on its own side


class StopRollingCanTable(StopRollingCanDirs):
    TASK_NAME = "StopRollingCanTable"
    START_DISTANCE = (0.70, 0.80)  # from the path's block point B, along the path (random)
    ANGLE_JITTER = 6.0  # deg, random around each direction
    # Direction each path comes from, measured at B from the robot's forward axis toward
    # its side (deg), and B (forward, |right|).
    DIRECTIONS = {"front": 25.0, "diag": 55.0, "side": 105.0}
    # How often each direction rolls (side cans, which only the side cameras see in time, half
    # of the episodes; front and diagonal a quarter each).
    DIRECTION_PROB = {"front": 0.25, "diag": 0.25, "side": 0.5}
    # Where each path ends (its block point B, robot frame forward, |right|), ahead of the
    # shoulder. (Spreading B over the front of the table did not stop a hand put there in
    # advance from blocking most paths, which all run toward the body, and put many B out of
    # the scripted policy's reach; FALSE_START below rules out moving early instead.)
    BLOCK_FORWARD = {"front": (0.52, 0.52), "diag": (0.48, 0.48), "side": (0.42, 0.42)}
    BLOCK_SIDE = {"front": (0.25, 0.25), "diag": (0.28, 0.28), "side": (0.30, 0.30)}
    # A hand may leave its resting place (palm moved more than this, m) only once the can
    # rolls; earlier is a false start and the episode fails. Without it, a hand put ahead of
    # the shoulder in advance blocks every path on that side without seeing anything.
    FALSE_START = 0.03
    # speed (m/s). At 0.50 a perception-limited oracle blocked 21% with the head camera, 46% with
    # the diagonal cameras added and 81% with the side cameras too (scripted policy: 83%).
    LEVELS = {3: dict(speed=0.45)}

    def _load_model(self, *args, **kwargs):
        Tabletop._load_model(self, *args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}table']") is None:
            self._add_scene()

    @classmethod
    def can_names(cls):
        return [f"{s}_{d}" for s in SIDES for d in cls.DIRECTIONS]

    def path(self, name):
        """(B, unit direction of travel) in robot-frame (forward, right) for the path `name`
        (side_direction); the episode's own path uses its sampled angle."""
        side, direction = name.split("_")
        sgn = SIDES[side]
        lay = getattr(self, "layout", None) or {}
        own = name == lay.get("can")
        th = np.deg2rad(lay["angle"] if own else self.DIRECTIONS[direction])
        bf = lay["block_forward"] if own else float(np.mean(self.BLOCK_FORWARD[direction]))
        br = lay["block_side"] if own else float(np.mean(self.BLOCK_SIDE[direction]))
        b = np.array([bf, sgn * br])
        return b, -np.array([np.cos(th), sgn * np.sin(th)])

    def _add_scene(self):
        self.layout = self._sample_layout()
        self.frame = self._frame()
        f0, f1 = TABLE_FORWARD
        self._box(self.model.worldbody, f"{PREFIX}table", self._world((f0 + f1) / 2, 0.0, -PLATFORM_HEIGHT / 2),
                  ((f1 - f0) / 2, TABLE_HALF_WIDTH, PLATFORM_HEIGHT / 2), COUNTER_RGBA, **SURFACE)
        self.can_start = {}
        b, d = self.path(self.target)
        p = b - self.layout["dist"] * d
        self._add_can(self.target, self._world(p[0], p[1], CAN_RADIUS))  # the only can

    def _sample_layout(self):
        if "can_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["can_layout"])
        side = str(self.rng.choice(list(SIDES)))
        dirs = list(self.DIRECTIONS)
        direction = str(self.rng.choice(dirs, p=[self.DIRECTION_PROB[d] for d in dirs]))
        return dict(can=f"{side}_{direction}", push_step=int(self.rng.integers(PUSH_STEPS[0], PUSH_STEPS[1] + 1)),
                    angle=float(self.DIRECTIONS[direction] + self.rng.uniform(-self.ANGLE_JITTER, self.ANGLE_JITTER)),
                    dist=float(self.rng.uniform(*self.START_DISTANCE)),
                    block_forward=float(self.rng.uniform(*self.BLOCK_FORWARD[direction])),
                    block_side=float(self.rng.uniform(*self.BLOCK_SIDE[direction])))

    def set_cameras(self):
        super().set_cameras()
        for spec in ROLL_CAMERAS:
            self._cam_configs[spec.name] = dict(
                pos=list(spec.pos), quat=list(spec.quat), parent_body=spec.parent_body,
                camera_attribs=dict(fovy=str(spec.fovy)),
            )

    def _reset_internal(self):
        super()._reset_internal()
        self._palm0 = {h: self._palm_pos(h) for h in SIDES}
        self._false_start = False

    def _palm_pos(self, hand):
        return self.sim.data.body_xpos[self.sim.model.body_name2id(PALMS[hand])].copy()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        if not self._pushed and not self._false_start:
            self._false_start = any(np.linalg.norm(self._palm_pos(h) - self._palm0[h]) > self.FALSE_START for h in SIDES)
        return reward, done, info

    def _check_success(self):
        _, rgt, _ = self._robot_frame(self.can_pos())
        own_side = SIDES[self.rolling_side] * rgt > MIDLINE_MARGIN
        return (self._pushed and not self._false_start and self._settled >= SETTLE_STEPS and self._can_on_counter()
                and own_side)

    def episode_record_header(self) -> dict:
        """Per-episode log header (registry GR1_MV_EPISODE_LOG_DIR)."""
        return dict(can=self.target, direction=self.direction, side=self.rolling_side, layout=dict(self.layout))

    def episode_record_step(self) -> dict:
        """Per-step log: can and palm positions (robot frame, mm), pushed / false-start flags."""
        mm = lambda p: [round(1000 * v) for v in self._robot_frame(p)]
        return dict(can=mm(self.can_pos()), palms={h: mm(self._palm_pos(h)) for h in SIDES},
                    pushed=self._pushed, false_start=self._false_start)

    def diagnostic_signals(self):
        return {"rolling_left": self.rolling_side == "left", "touched": self._touched, "false_start": self._false_start,
                **{f"from_{d}": self.direction == d for d in self.DIRECTIONS}}


ENV_NAMES = make_level_classes(StopRollingCanTable, __name__)
