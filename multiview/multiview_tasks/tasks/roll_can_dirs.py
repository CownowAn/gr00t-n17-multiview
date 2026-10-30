"""Task 9 (prototype): Stop the Can Rolling in from Any Direction (field of view, reaction).

The work surface wraps around the robot: a wide counter in front and two wings beside
its body that reach back past it. Six identical soda cans lie at its edges, three per
side, each waiting at the end of a straight path that runs past a block point B ahead of
that side's shoulder: one comes in obliquely from ahead ("front"), one straight from the
side ("side") and one from behind the side ("back"). At a random moment one of the six
is pushed and rolls along its path at a fixed speed; unstopped it rolls on into the
midline gutter. The hand on that side has to get into its path and stop it: "stop the
rolling can".

What has to be perceived: which can rolls (only the motion tells) and when. The head
camera looks forward and down, so it sees a can from ahead start rolling but one from
the side or behind only shortly before it arrives. Cameras looking straight out to the
sides see the side cans start, cameras looking back past the shoulders the back cans.
The hands are slow (about 0.5 s per 10 cm), so seeing a can late leaves no time. The
waist is locked, so the robot cannot turn to look.

Knob: the can speed.

Success: the rolling can stays on the counter and stops moving along its path (it may
rest against the hand or roll slowly back).
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import Tabletop
from robosuite.utils.mjcf_utils import array_to_string

from multiview_tasks.tasks.catch_ball import SIDES, CatchRollingBall
from multiview_tasks.tasks.roll_can import (
    CAN_HALF_LENGTH, CAN_MASS, CAN_RADIUS, CAN_RGBA, CAN_RIM_RGBA, COUNTER_RGBA, PLATFORM_HEIGHT, PUSH_STEPS,
    SETTLE_SPEED, SETTLE_STEPS, SURFACE,
)
from multiview_tasks.variant import make_level_classes

PREFIX = "mvcand_"

# Work surface (robot frame: forward, right; metres from the robot base), raised as in the
# other multi-view tasks. Front counter split along the midline by a gutter; wings beside
# the body (clear of the resting hands, which stay within 0.26 m of the midline).
FRONT_FORWARD = (0.18, 1.25)
FRONT_SIDE = (0.06, 1.20)  # |right| extent of each half of the front counter
WING_FORWARD = (-0.48, 0.18)
WING_SIDE = (0.42, 1.20)

START_DISTANCE = 0.75  # from the path's block point B, along the path
# Each path: the direction it comes from, measured at its block point B from the robot's
# forward axis toward its side, and B (forward, |right|). All end in the midline gutter,
# well ahead of the resting hands (which reach about 0.3 m forward).
DIRECTIONS = {"front": 50.0, "side": 90.0, "back": 145.0}
BLOCK_POINTS = {"front": (0.55, 0.25), "side": (0.42, 0.30), "back": (0.42, 0.30)}


class StopRollingCanDirs(CatchRollingBall):
    TASK_NAME = "StopRollingCanDirs"
    LEVELS = {
        0: dict(speed=0.40),
        1: dict(speed=0.45),
        2: dict(speed=0.50),
        3: dict(speed=0.55),
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        super().__init__(*args, **kwargs)

    def _load_model(self, *args, **kwargs):
        Tabletop._load_model(self, *args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}can_right_side']") is None:
            self._add_scene()

    @staticmethod
    def can_names():
        return [f"{s}_{d}" for s in SIDES for d in DIRECTIONS]

    def _sample_layout(self):
        if "can_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["can_layout"])
        return dict(
            can=str(self.rng.choice(self.can_names())),
            push_step=int(self.rng.integers(PUSH_STEPS[0], PUSH_STEPS[1] + 1)),
        )

    def path(self, name):
        """(B, unit direction of travel) in robot-frame (forward, right) for can `name`."""
        side, direction = name.split("_")
        sgn = SIDES[side]
        th = np.deg2rad(DIRECTIONS[direction])
        b = np.array([BLOCK_POINTS[direction][0], sgn * BLOCK_POINTS[direction][1]])
        return b, -np.array([np.cos(th), sgn * np.sin(th)])

    def _add_scene(self):
        self.layout = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        for side, sgn in SIDES.items():
            for part, (f0, f1), (s0, s1) in (("front", FRONT_FORWARD, FRONT_SIDE), ("wing", WING_FORWARD, WING_SIDE)):
                self._box(wb, f"{PREFIX}{part}_{side}", self._world((f0 + f1) / 2, sgn * (s0 + s1) / 2, -PLATFORM_HEIGHT / 2),
                          ((f1 - f0) / 2, (s1 - s0) / 2, PLATFORM_HEIGHT / 2), COUNTER_RGBA, **SURFACE)
        self.can_start = {}
        for name in self.can_names():
            b, d = self.path(name)
            p = b - START_DISTANCE * d
            self._add_can(name, self._world(p[0], p[1], CAN_RADIUS))

    def _can_quat(self, name):
        """wxyz: the can's axis horizontal, across its path."""
        _, d = self.path(name)
        heading = np.rad2deg(np.arctan2(d[1], d[0]))  # toward the robot's right from forward
        return self._quat(pitch_deg=90.0, turn_deg=heading + 90.0)

    def _add_can(self, name, start):
        self.can_start[name] = start
        body = f"{PREFIX}can_{name}"
        can = ET.SubElement(self.model.worldbody, "body", name=body, pos=array_to_string(start),
                            quat=array_to_string(self._can_quat(name)))
        ET.SubElement(can, "freejoint", name=f"{body}_joint")
        r, h, m = CAN_RADIUS, CAN_HALF_LENGTH, CAN_MASS
        i_axis, i_side = 0.5 * m * r**2, m * (3 * r**2 + (2 * h) ** 2) / 12
        ET.SubElement(can, "inertial", pos="0 0 0", mass=str(m), diaginertia=f"{i_side} {i_side} {i_axis}")
        ET.SubElement(can, "geom", name=f"{body}_geom", type="cylinder", size=f"{r} {h}", rgba=CAN_RGBA,
                      group="1", **SURFACE)
        for k, z in enumerate((-h, h)):
            ET.SubElement(can, "geom", name=f"{body}_rim{k}", type="cylinder", size=f"{r * 1.005} 0.004",
                          pos=f"0 0 {z}", rgba=CAN_RIM_RGBA, contype="0", conaffinity="0", group="1")

    # ------------------------------------------------------------------
    # Reset, push
    # ------------------------------------------------------------------
    def _reset_internal(self):
        Tabletop._reset_internal(self)
        model, data = self.sim.model._model, self.sim.data._data
        self._can_adr = {}
        for name, start in self.can_start.items():
            j = model.joint(f"{PREFIX}can_{name}_joint")
            qa, da = model.jnt_qposadr[j.id], model.jnt_dofadr[j.id]
            self._can_adr[name] = (qa, da)
            data.qpos[qa:qa + 7] = [*start, *self._can_quat(name)]
            data.qvel[da:da + 6] = 0.0
        self._can_body = model.body(f"{PREFIX}can_{self.target}").id
        self._hand_bodies = {h: {b for b in range(model.nbody)
                                 if model.body(b).name.startswith((f"gripper0_{h}", f"robot0_{h[0]}_upper_arm",
                                                                   f"robot0_{h[0]}_lower_arm", f"robot0_{h[0]}_hand"))}
                             for h in SIDES}
        self._step = 0
        self._pushed = False
        self._touched = False
        self._settled = 0
        self.sim.forward()

    @property
    def target(self) -> str:
        return self.layout["can"]

    @property
    def rolling_side(self) -> str:
        return self.target.split("_")[0]

    @property
    def direction(self) -> str:
        return self.target.split("_")[1]

    def can_pos(self, name=None) -> np.ndarray:
        return self.sim.data.body_xpos[self.sim.model.body_name2id(f"{PREFIX}can_{name or self.target}")].copy()

    def can_speed(self, name=None) -> float:
        _, da = self._can_adr[name or self.target]
        return float(np.linalg.norm(self.sim.data.qvel[da:da + 3]))

    def _travel_world(self):
        _, d = self.path(self.target)
        _, _, fwd, rgt, _ = self.frame
        return d[0] * fwd + d[1] * rgt

    def _along_speed(self) -> float:
        _, da = self._can_adr[self.target]
        return float(self.sim.data.qvel[da:da + 3] @ self._travel_world())

    def _push(self):
        vel = self.variant_params["speed"] * self._travel_world()
        omega = np.cross(np.array([0.0, 0.0, 1.0]), vel) / CAN_RADIUS
        _, da = self._can_adr[self.target]
        rot = self.sim.data.body_xmat[self._can_body].reshape(3, 3)
        self.sim.data._data.qvel[da:da + 6] = [*vel, *(rot.T @ omega)]  # a free joint's angular velocity is in the body frame

    def _hand_touching(self) -> bool:
        model, data = self.sim.model._model, self.sim.data._data
        hand = self._hand_bodies[self.rolling_side]
        for i in range(data.ncon):
            b1, b2 = model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]
            if (b1 == self._can_body and b2 in hand) or (b2 == self._can_body and b1 in hand):
                return True
        return False

    def _post_action(self, action):
        reward, done, info = Tabletop._post_action(self, action)
        self._step += 1
        if not self._pushed and self._step >= self.layout["push_step"]:
            self._push()
            self._pushed = True
        elif self._pushed:
            self._touched = self._touched or self._hand_touching()
            self._settled = self._settled + 1 if self._along_speed() < SETTLE_SPEED else 0
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def _can_on_counter(self) -> bool:
        return self._robot_frame(self.can_pos())[2] > CAN_RADIUS - 0.01

    def _check_success(self):
        return self._pushed and self._settled >= SETTLE_STEPS and self._can_on_counter()

    def get_ep_meta(self):
        ep_meta = Tabletop.get_ep_meta(self)
        if hasattr(self, "layout"):
            ep_meta["lang"] = "stop the rolling can"
            ep_meta["can_layout"] = dict(self.layout)
        return ep_meta

    def phase_signals(self):
        return dict(pushed=self._pushed, stopped=self._check_success())

    def diagnostic_signals(self):
        return {"rolling_left": self.rolling_side == "left", "touched": self._touched,
                **{f"from_{d}": self.direction == d for d in DIRECTIONS}}


ENV_NAMES = make_level_classes(StopRollingCanDirs, __name__)
