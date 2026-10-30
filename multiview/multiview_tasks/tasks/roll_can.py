"""Task 8 (prototype): Stop the Rolling Can (field of view, reaction).

A long counter runs across in front of the robot and out to both sides. A soda can lies
on its side at the far end of each side, its axis pointing forward. At a random moment
one of the two cans (random side) is pushed and rolls across the counter toward the
midline at a fixed speed; unstopped, it drops into the gutter along the midline. The
hand on that side has to get into its path and stop it: "stop the rolling can".

What has to be perceived: which can rolls, along which line (the cans wait at a random
distance from the robot) and when it starts (random over 3.5 s). The cans look the same
and both wait at the ends, so only the motion tells which one rolls. The head camera
looks forward and down at the work area, so a can coming in from the side enters its
image only shortly before it arrives; cameras that look out to the sides see it start.
The hands are slow (about 0.5 s per 10 cm), so seeing the can late leaves no time to get
a hand in front of it. The waist is locked, so the robot cannot turn to look.

Knob: the speed of the can (faster, the head camera's warning gets shorter).

Success: the rolling can stays on its own side of the gutter and stops moving toward the
midline (it may rest against the hand or roll slowly back out). Unstopped, it keeps
rolling into the gutter (rolling friction alone takes off only ~0.04 m/s per metre).
A bounce off the hand can begin and end within one control step, so contact with the
hand is recorded for diagnostics but not required.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import Tabletop
from robosuite.utils.mjcf_utils import array_to_string

from multiview_tasks.tasks import catch_ball as cb
from multiview_tasks.tasks.catch_ball import SIDES, CatchRollingBall
from multiview_tasks.variant import make_level_classes

PREFIX = "mvcan_"

# Counter (robot frame: forward, right; metres from the robot base), raised as in the
# other multi-view tasks; split along the midline by a gutter.
PLATFORM_HEIGHT = cb.PLATFORM_HEIGHT
PLATFORM_FORWARD = (0.18, 0.66)
PLATFORM_SIDE = (0.06, 1.15)  # |right| extent of each half; the gap between the halves is the gutter
COUNTER_RGBA = "0.62 0.52 0.40 1"
# Low back wall along the far edge (keeps a deflected can on the counter).
BACKSPLASH_HEIGHT = 0.06

# Contact parameters of the counter and the cans: condim 6 adds rolling friction.
SURFACE = dict(friction="1 0.005 0.0001", condim="6")

CAN_RADIUS = 0.033
CAN_HALF_LENGTH = 0.061
CAN_MASS = 0.10  # an empty can (a full one, 0.37 kg, would take the fingers with it)
CAN_RGBA = "0.80 0.10 0.12 1"
CAN_RIM_RGBA = "0.78 0.78 0.80 1"

CAN_START_SIDE = 1.02  # |right| of a waiting can
LANE_RANGE = (0.36, 0.52)  # forward position of each can's path
PUSH_STEPS = (10, 80)  # the can is pushed at a random step in this range (20 Hz)
SETTLE_STEPS = 10  # once pushed, the can must not move toward the midline for this many steps
SETTLE_SPEED = 0.03  # m/s toward the midline still counted as stopped (pressed against the hand it jitters)


class StopRollingCan(CatchRollingBall):
    TASK_NAME = "StopRollingCan"
    # speed: of the pushed can (m/s).
    LEVELS = {
        0: dict(speed=0.25),
        1: dict(speed=0.30),
        2: dict(speed=0.35),
        3: dict(speed=0.40),
    }

    def _load_model(self, *args, **kwargs):
        Tabletop._load_model(self, *args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}can_right']") is None:
            self._add_scene()

    def _sample_layout(self):
        if "can_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["can_layout"])
        lanes = self.rng.uniform(*LANE_RANGE, size=2)
        return dict(
            lane_left=float(lanes[0]), lane_right=float(lanes[1]),
            side=str(self.rng.choice(list(SIDES))),
            push_step=int(self.rng.integers(PUSH_STEPS[0], PUSH_STEPS[1] + 1)),
        )

    def _add_scene(self):
        self.layout = lay = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        f0, f1 = PLATFORM_FORWARD
        s0, s1 = PLATFORM_SIDE
        self.can_start = {}
        for side, sgn in SIDES.items():
            self._box(wb, f"{PREFIX}counter_{side}", self._world((f0 + f1) / 2, sgn * (s0 + s1) / 2, -PLATFORM_HEIGHT / 2),
                      ((f1 - f0) / 2, (s1 - s0) / 2, PLATFORM_HEIGHT / 2), COUNTER_RGBA, **SURFACE)
            self._box(wb, f"{PREFIX}backsplash_{side}", self._world(f1 + 0.01, sgn * (s0 + s1) / 2, BACKSPLASH_HEIGHT / 2),
                      (0.01, (s1 - s0) / 2, BACKSPLASH_HEIGHT / 2), COUNTER_RGBA)
            start = self._world(lay[f"lane_{side}"], sgn * CAN_START_SIDE, CAN_RADIUS)
            self._add_can(side, start)

    def _add_can(self, side, start):
        """A can lying with its axis along the robot's forward direction (it rolls sideways)."""
        self.can_start[side] = start
        name = f"{PREFIX}can_{side}"
        can = ET.SubElement(self.model.worldbody, "body", name=name, pos=array_to_string(start),
                            quat=array_to_string(self._quat(pitch_deg=90.0)))  # local z (the axis) -> forward
        ET.SubElement(can, "freejoint", name=f"{name}_joint")
        r, h, m = CAN_RADIUS, CAN_HALF_LENGTH, CAN_MASS
        i_axis, i_side = 0.5 * m * r**2, m * (3 * r**2 + (2 * h) ** 2) / 12  # solid cylinder
        ET.SubElement(can, "inertial", pos="0 0 0", mass=str(m), diaginertia=f"{i_side} {i_side} {i_axis}")
        ET.SubElement(can, "geom", name=f"{name}_geom", type="cylinder", size=f"{r} {h}", rgba=CAN_RGBA,
                      group="1", **SURFACE)
        for k, z in enumerate((-h, h)):  # silver rims (visual only)
            ET.SubElement(can, "geom", name=f"{name}_rim{k}", type="cylinder", size=f"{r * 1.005} 0.004",
                          pos=f"0 0 {z}", rgba=CAN_RIM_RGBA, contype="0", conaffinity="0", group="1")

    # ------------------------------------------------------------------
    # Reset, push
    # ------------------------------------------------------------------
    def _reset_internal(self):
        Tabletop._reset_internal(self)
        model, data = self.sim.model._model, self.sim.data._data
        q = self._quat(pitch_deg=90.0)
        self._can_adr = {}
        for side, start in self.can_start.items():
            j = model.joint(f"{PREFIX}can_{side}_joint")
            self._can_adr[side] = (model.jnt_qposadr[j.id], model.jnt_dofadr[j.id])
            qa, da = self._can_adr[side]
            data.qpos[qa:qa + 7] = [*start, *q]
            data.qvel[da:da + 6] = 0.0
        self._can_bodies = {model.body(f"{PREFIX}can_{s}").id: s for s in SIDES}
        self._hand_bodies = {h: {b for b in range(model.nbody)  # the hand and its arm (a forearm may block too)
                                 if model.body(b).name.startswith((f"gripper0_{h}", f"robot0_{h[0]}_upper_arm",
                                                                   f"robot0_{h[0]}_lower_arm", f"robot0_{h[0]}_hand"))}
                             for h in SIDES}
        self._step = 0
        self._pushed = False
        self._touched = False
        self._settled = 0  # steps in a row the pushed can has not moved toward the midline
        self.sim.forward()

    @property
    def rolling_side(self) -> str:
        return self.layout["side"]

    def can_pos(self, side=None) -> np.ndarray:
        return self.sim.data.body_xpos[self.sim.model.body_name2id(f"{PREFIX}can_{side or self.rolling_side}")].copy()

    def can_speed(self, side=None) -> float:
        _, da = self._can_adr[side or self.rolling_side]
        return float(np.linalg.norm(self.sim.data.qvel[da:da + 3]))

    def _push(self):
        """Set the rolling can moving toward the midline, rolling without slipping."""
        side = self.rolling_side
        _, _, forward, right, _ = self.frame
        v = self.variant_params["speed"]
        vel = -SIDES[side] * v * right  # toward the midline
        omega = np.cross(np.array([0.0, 0.0, 1.0]), vel) / CAN_RADIUS  # rolling: v = omega x (r up)
        _, da = self._can_adr[side]
        rot = self.sim.data.body_xmat[self.sim.model.body_name2id(f"{PREFIX}can_{side}")].reshape(3, 3)
        self.sim.data._data.qvel[da:da + 6] = [*vel, *(rot.T @ omega)]  # a free joint's angular velocity is in the body frame

    def _hand_touching(self) -> bool:
        model, data = self.sim.model._model, self.sim.data._data
        hand = self._hand_bodies[self.rolling_side]
        for i in range(data.ncon):
            b1, b2 = model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]
            for a, b in ((b1, b2), (b2, b1)):
                if self._can_bodies.get(a) == self.rolling_side and b in hand:
                    return True
        return False

    def _post_action(self, action):
        reward, done, info = Tabletop._post_action(self, action)
        self._step += 1
        if not self._pushed and self._step >= self.layout["push_step"]:
            self._push()
            self._pushed = True
        if self._pushed:
            self._touched = self._touched or self._hand_touching()
            self._settled = self._settled + 1 if self._inward_speed() < SETTLE_SPEED else 0
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def _can_on_own_side(self) -> bool:
        fwd, rgt, z = self._robot_frame(self.can_pos())
        return z > CAN_RADIUS - 0.01 and SIDES[self.rolling_side] * rgt > PLATFORM_SIDE[0]

    def _check_success(self):
        return self._pushed and self._settled >= SETTLE_STEPS and self._can_on_own_side()

    def _inward_speed(self) -> float:
        """Speed of the rolling can toward the midline (negative: rolling back out)."""
        _, da = self._can_adr[self.rolling_side]
        _, _, _, right, _ = self.frame
        return float(-SIDES[self.rolling_side] * (self.sim.data.qvel[da:da + 3] @ right))

    def get_ep_meta(self):
        ep_meta = Tabletop.get_ep_meta(self)
        if hasattr(self, "layout"):
            ep_meta["lang"] = "stop the rolling can"
            ep_meta["can_layout"] = dict(self.layout)
        return ep_meta

    def phase_signals(self):
        return dict(pushed=self._pushed, stopped=self._check_success())

    def diagnostic_signals(self):
        return dict(rolling_left=self.rolling_side == "left", touched=self._touched,
                    in_gutter=self._pushed and not self._can_on_own_side())


ENV_NAMES = make_level_classes(StopRollingCan, __name__)
