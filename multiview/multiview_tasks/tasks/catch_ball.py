"""Task 5 (prototype): Catch the Rolling Ball (temporal occlusion by low canopies).

Two straight tracks run toward the robot, one on each side, on a raised work surface.
At the far end of each track a ball waits on a wide ramp. At a random moment one of the
two balls (random side) is let go, rolls down the ramp and along its track at a fixed
speed (~0.3 m/s) and would roll off the surface's near edge. The hand on that side has
to get into the ball's lane and close on the ball as it passes: "catch the rolling ball".

What has to be perceived: which ball rolls, where its lane is (random over 25 cm on each
side) and when it starts (random over 3.5 s). The balls look the same and both wait on
their ramps, so only the motion tells which one rolls. The speed is fixed, so one frame
showing the rolling ball tells where it will cross the catch line and when. The hands
are slow (about 0.5 s per 10 cm), so seeing the ball late leaves no time to move the
hand to a far lane.

Knob: a low canopy over the far part of both tracks. A camera high above the surface
looks down onto the canopy and sees the ball only once it rolls out from under it; a
camera mounted lower on the torso looks under it. The longer the canopy, the later the
head camera sees the ball (L0: no canopy). The waist is locked.

Catching: a weld holds the ball (GR1's hand cannot hold it by friction alone here). The
weld forms only while a hand is closing: within GRASP_WINDOW_STEPS env steps after that
hand's mean finger flexion rises past GRASP_FINGER_ANGLE (from open, below
RELEASE_FINGER_ANGLE), a ball's center has to come within GRASP_DISTANCE of its grasp
point. A hand that is already closed does not catch, and the caught ball has to be the
rolling one and still rolling.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.variant import VariantMixin, make_level_classes

PREFIX = "mvcatch_"
SIDES = {"left": -1.0, "right": 1.0}  # sign of the robot-frame "right" coordinate

# Raised work surface (robot frame: forward, right; metres from the robot base).
PLATFORM_HEIGHT = 0.22
PLATFORM_FORWARD = (0.18, 1.00)
PLATFORM_HALF_WIDTH = 0.52

# Contact parameters of the rolling surfaces and the balls: condim 6 adds rolling
# friction (with the default condim 3 a ball never slows down).
SURFACE = dict(friction="1 0.005 0.0001", condim="6")

BALL_RADIUS = 0.02
BALL_MASS = 0.05
BALL_RGBA = "0.95 0.45 0.05 1"

# Lanes: |right| of the lane's center line on each side; a palm-down hand with the
# fingers turned inward reaches table height over this range at the catch line.
LANE_RANGE = (0.15, 0.40)
CATCH_FORWARD = 0.42  # where the scripted hand meets the ball
# Ramps: the low edge meets the surface at RAMP_FORWARD; a ball starts RAMP_START up the
# slope, which sets the speed (~0.3 m/s off the ramp).
RAMP_FORWARD = 0.80
RAMP_DEG = 5.0
RAMP_LENGTH = 0.18
RAMP_START = 0.10
RAMP_SPAN = (0.11, 0.44)  # |right| extent of each ramp
# Let-go time after reset (env steps at 20 Hz).
RELEASE_STEPS = (10, 80)

# Canopy: one roof over both tracks from the level's canopy_near to the far end, its
# underside CANOPY_HEIGHT above the surface, held by walls along the outer edges. It
# spans the midline, so a camera on the torso looks under it only through its open
# near end, and how far it sees in depends on its height.
CANOPY_HEIGHT = 0.06  # 1 cm over a ball on the ramp: a camera above it sees little of the ball underneath
CANOPY_HALF_WIDTH = 0.47
CANOPY_THICKNESS = 0.01
CANOPY_FAR = 1.00
CANOPY_RGBA = "0.45 0.55 0.62 1"

# Catching (see the module docstring).
PALMS = {"left": "gripper0_left_l_palm", "right": "gripper0_right_r_palm"}
GRASP_POINT_IN_PALM = {  # where the closed fingertips meet, in each palm frame
    "right": np.array([0.028, 0.0464, -0.1002]),
    "left": np.array([0.028, -0.0464, -0.1002]),
}
GRASP_FINGER_ANGLE = 0.35
RELEASE_FINGER_ANGLE = 0.3
GRASP_DISTANCE = 0.06  # 0.05 left a 2-step closing window for a 2.5 cm ball (the closing fingers stop it short)
GRASP_WINDOW_STEPS = 4  # a step-closed hand passes 0.35 -> ~0.6 in this time
MIN_CATCH_SPEED = 0.1  # m/s; the ball must still be rolling (fastest over the last ROLLING_STEPS steps)
ROLLING_STEPS = 6  # a ball stopped by the fingers a moment before the weld forms still counts
HOLD_SOLREF = "0.004 1"
LIFT_HEIGHT = 0.05  # success: the caught ball this far above its resting height


class CatchRollingBall(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "CatchRollingBall"
    VALID_LAYOUTS = [0]
    LOCK_WAIST = True
    GRASP_WINDOW_STEPS = GRASP_WINDOW_STEPS
    GRASP_ON_CONTACT = False  # also catch a ball that touches the closing hand (not only one within GRASP_DISTANCE)
    # canopy_near: forward position of the canopies' near edge (m); None = no canopy.
    LEVELS = {
        0: dict(canopy_near=None),
        1: dict(canopy_near=0.70),
        2: dict(canopy_near=0.60),
        3: dict(canopy_near=0.52),
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55)))
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        return []

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}ball_right']") is None:
            self._add_scene()

    def _sample_layout(self):
        if "catch_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["catch_layout"])
        lanes = self.rng.uniform(*LANE_RANGE, size=2)
        return dict(
            lane_left=float(lanes[0]), lane_right=float(lanes[1]),
            side=str(self.rng.choice(list(SIDES))),
            release_step=int(self.rng.integers(RELEASE_STEPS[0], RELEASE_STEPS[1] + 1)),
        )

    def _frame(self):
        base_pos, base_ori = self.compute_robot_base_placement_pose()
        yaw = base_ori[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        top = self.counter.get_ext_sites(relative=False)[3][2] + PLATFORM_HEIGHT
        return base_pos, yaw, forward, right, top

    def _world(self, fwd, rgt, z):
        """Robot-frame (forward, right, height above the work surface) -> world."""
        base_pos, _, forward, right, top = self.frame
        p = base_pos + fwd * forward + rgt * right
        return np.array([p[0], p[1], top + z])

    def _quat(self, pitch_deg=0.0, turn_deg=0.0):
        """wxyz of the robot's yaw turned `turn_deg` toward the robot's right, then a pitch about the
        local lateral axis (raises local +x for > 0)."""
        _, yaw, _, _, _ = self.frame
        r = Rotation.from_euler("z", yaw - np.deg2rad(turn_deg)) * Rotation.from_euler("y", -pitch_deg, degrees=True)
        qx, qy, qz, qw = r.as_quat()
        return [qw, qx, qy, qz]

    def _box(self, parent, name, center, half, rgba, pitch_deg=0.0, turn_deg=0.0, **geom_kw):
        body = ET.SubElement(parent, "body", name=name, pos=array_to_string(center),
                             quat=array_to_string(self._quat(pitch_deg, turn_deg)))
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=array_to_string(half),
                      rgba=rgba, group="1", **geom_kw)
        return body

    def _add_scene(self):
        self.layout = lay = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        f0, f1 = PLATFORM_FORWARD
        self._box(wb, f"{PREFIX}platform", self._world((f0 + f1) / 2, 0.0, -PLATFORM_HEIGHT / 2),
                  ((f1 - f0) / 2, PLATFORM_HALF_WIDTH, PLATFORM_HEIGHT / 2), "0.62 0.52 0.40 1", **SURFACE)
        a = np.deg2rad(RAMP_DEG)
        t = 0.004  # ramp plank half thickness
        mid = RAMP_LENGTH / 2
        hi = RAMP_LENGTH * np.sin(a)
        canopy_near = self.variant_params["canopy_near"]
        self.ball_start = {}
        for side, sgn in SIDES.items():
            r_mid = sgn * (RAMP_SPAN[0] + RAMP_SPAN[1]) / 2
            r_half = (RAMP_SPAN[1] - RAMP_SPAN[0]) / 2
            self._box(wb, f"{PREFIX}ramp_{side}",
                      self._world(RAMP_FORWARD + mid * np.cos(a) + t * np.sin(a), r_mid, mid * np.sin(a) - t * np.cos(a)),
                      (mid, r_half, t), "0.80 0.80 0.76 1", pitch_deg=RAMP_DEG, **SURFACE)
            self._box(wb, f"{PREFIX}ramp_post_{side}",
                      self._world(RAMP_FORWARD + RAMP_LENGTH * np.cos(a) + 0.01, r_mid, hi / 2),
                      (0.01, r_half, hi / 2), "0.80 0.80 0.76 1")
            if canopy_near is not None:
                f_mid, f_half = (canopy_near + CANOPY_FAR) / 2, (CANOPY_FAR - canopy_near) / 2
                self._box(wb, f"{PREFIX}canopy_wall_{side}",
                          self._world(f_mid, sgn * (CANOPY_HALF_WIDTH + CANOPY_THICKNESS / 2), CANOPY_HEIGHT / 2),
                          (f_half, CANOPY_THICKNESS / 2, CANOPY_HEIGHT / 2), CANOPY_RGBA)
            # Ball center: RAMP_START up the slope, one radius off it along the surface normal.
            start = self._world(RAMP_FORWARD + RAMP_START * np.cos(a) - BALL_RADIUS * np.sin(a), sgn * lay[f"lane_{side}"],
                                RAMP_START * np.sin(a) + BALL_RADIUS * np.cos(a))
            self._add_ball(side, start)
        if canopy_near is not None:
            f_mid, f_half = (canopy_near + CANOPY_FAR) / 2, (CANOPY_FAR - canopy_near) / 2
            self._box(wb, f"{PREFIX}canopy", self._world(f_mid, 0.0, CANOPY_HEIGHT + CANOPY_THICKNESS / 2),
                      (f_half, CANOPY_HALF_WIDTH + CANOPY_THICKNESS, CANOPY_THICKNESS / 2), CANOPY_RGBA)
        self._add_grasp_welds()

    def _add_ball(self, side, start, radius=BALL_RADIUS):
        """A ball waiting at `start` (held by a weld to the world until it is let go)."""
        self.ball_start[side] = start
        name = f"{PREFIX}ball_{side}"
        ball = ET.SubElement(self.model.worldbody, "body", name=name, pos=array_to_string(start))
        ET.SubElement(ball, "freejoint", name=f"{name}_joint")
        inertia = 0.4 * BALL_MASS * radius ** 2  # solid sphere (the scene does not infer inertia from geoms)
        ET.SubElement(ball, "inertial", pos="0 0 0", mass=str(BALL_MASS), diaginertia=f"{inertia} {inertia} {inertia}")
        ET.SubElement(ball, "geom", name=f"{name}_geom", type="sphere", size=str(radius),
                      rgba=BALL_RGBA, group="1", **SURFACE)
        ET.SubElement(self.model.equality, "weld", name=f"{name}_wait", body1="world", body2=name, active="true")

    def _add_grasp_welds(self):
        """A (disabled) weld from each palm to each ball, and the grasp point sites."""
        for hand, palm in PALMS.items():
            for side in SIDES:
                ET.SubElement(self.model.equality, "weld", name=f"{PREFIX}{hand}_holds_{side}", body1=palm,
                              body2=f"{PREFIX}ball_{side}", active="false", solref=HOLD_SOLREF)
            palm_el = self.model.worldbody.find(f".//body[@name='{palm}']")
            ET.SubElement(palm_el, "site", name=f"{PREFIX}{hand}_grasp", pos=array_to_string(GRASP_POINT_IN_PALM[hand]),
                          size="0.005", group="5")

    # ------------------------------------------------------------------
    # Reset, release, catching
    # ------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        model, data = self.sim.model._model, self.sim.data._data
        self._wait_eq, self._hold_eq = {}, {}
        for side, start in self.ball_start.items():
            j = model.joint(f"{PREFIX}ball_{side}_joint")
            data.qpos[model.jnt_qposadr[j.id]:model.jnt_qposadr[j.id] + 7] = [*start, 1, 0, 0, 0]
            data.qvel[model.jnt_dofadr[j.id]:model.jnt_dofadr[j.id] + 6] = 0.0
            eq = model.equality(f"{PREFIX}ball_{side}_wait").id
            model.eq_data[eq, :] = 0.0
            model.eq_data[eq, 3:6] = start  # body1 = world: the ball's pose in the world frame
            model.eq_data[eq, 6] = 1.0
            model.eq_data[eq, 10] = 1.0
            data.eq_active[eq] = 1
            self._wait_eq[side] = eq
            for hand in PALMS:
                self._hold_eq[hand, side] = model.equality(f"{PREFIX}{hand}_holds_{side}").id
                data.eq_active[self._hold_eq[hand, side]] = 0
        self._step = 0
        self._released = False
        self._held = {hand: None for hand in PALMS}  # side of the ball each hand holds
        self._armed = {hand: False for hand in PALMS}  # hand open since its last closing
        self._close_step = {hand: None for hand in PALMS}  # step its flexion last rose past the grasp angle
        self._caught = False
        self._catch_speed = None
        self._speeds = {side: [] for side in SIDES}  # recent speeds of each ball
        self.sim.forward()

    def _set_weld(self, eq, b1, b2):
        model, data = self.sim.model._model, self.sim.data
        r1 = data.body_xmat[b1].reshape(3, 3)
        r2 = data.body_xmat[b2].reshape(3, 3)
        rel = Rotation.from_matrix(r1.T @ r2).as_quat()
        model.eq_data[eq, 0:3] = 0.0
        model.eq_data[eq, 3:6] = r1.T @ (data.body_xpos[b2] - data.body_xpos[b1])
        model.eq_data[eq, 6:10] = [rel[3], rel[0], rel[1], rel[2]]
        model.eq_data[eq, 10] = 1.0
        data._data.eq_active[eq] = 1

    @property
    def rolling_side(self) -> str:
        return self.layout["side"]

    def finger_flexion(self, hand) -> float:
        q = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes[hand]]
        return float(np.mean(q[:8]))  # pinky, ring, middle and index joints

    def grasp_point(self, hand) -> np.ndarray:
        return self.sim.data.site_xpos[self.sim.model.site_name2id(f"{PREFIX}{hand}_grasp")].copy()

    def ball_pos(self, side=None) -> np.ndarray:
        return self.sim.data.body_xpos[self.sim.model.body_name2id(f"{PREFIX}ball_{side or self.rolling_side}")].copy()

    def ball_speed(self, side=None) -> float:
        model = self.sim.model._model
        dof = model.jnt_dofadr[model.joint(f"{PREFIX}ball_{side or self.rolling_side}_joint").id]
        return float(np.linalg.norm(self.sim.data.qvel[dof:dof + 3]))

    def _balls_touching(self, hand) -> set:
        """Sides of the balls in contact with that hand's fingers or palm."""
        model, data = self.sim.model._model, self.sim.data._data
        if not hasattr(self, "_hand_bodies"):
            self._hand_bodies = {h: {b for b in range(model.nbody) if model.body(b).name.startswith(f"gripper0_{h}")}
                                 for h in PALMS}
            self._ball_bodies = {model.body(f"{PREFIX}ball_{s}").id: s for s in SIDES}
        sides = set()
        for i in range(data.ncon):
            b1, b2 = model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]
            for a, b in ((b1, b2), (b2, b1)):
                if a in self._ball_bodies and b in self._hand_bodies[hand]:
                    sides.add(self._ball_bodies[a])
        return sides

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        data = self.sim.data._data
        self._step += 1
        if not self._released and self._step >= self.layout["release_step"]:
            data.eq_active[self._wait_eq[self.rolling_side]] = 0
            self._released = True
        for side in SIDES:
            self._speeds[side] = (self._speeds[side] + [self.ball_speed(side)])[-ROLLING_STEPS:]
        for hand, palm in PALMS.items():
            flex = self.finger_flexion(hand)
            if self._held[hand] is not None:
                if flex < RELEASE_FINGER_ANGLE:
                    data.eq_active[self._hold_eq[hand, self._held[hand]]] = 0
                    self._held[hand] = None
                continue
            if flex < RELEASE_FINGER_ANGLE:
                self._armed[hand], self._close_step[hand] = True, None
            elif flex > GRASP_FINGER_ANGLE and self._armed[hand]:
                self._armed[hand], self._close_step[hand] = False, self._step
            closing = self._close_step[hand] is not None and self._step - self._close_step[hand] <= self.GRASP_WINDOW_STEPS
            if not closing:
                continue
            point = self.grasp_point(hand)
            held_elsewhere = {s for s in self._held.values() if s is not None}
            dist = {s: np.linalg.norm(self.ball_pos(s) - point) for s in SIDES if s not in held_elsewhere}
            if self.GRASP_ON_CONTACT:
                for s in self._balls_touching(hand):
                    if s in dist:
                        dist[s] = 0.0
            side = min(dist, key=dist.get) if dist else None
            if side is not None and dist[side] < GRASP_DISTANCE:
                speed = max(self._speeds[side])
                self._set_weld(self._hold_eq[hand, side], self.sim.model.body_name2id(palm),
                               self.sim.model.body_name2id(f"{PREFIX}ball_{side}"))
                self._held[hand], self._close_step[hand] = side, None
                if side == self.rolling_side and self._released and speed > MIN_CATCH_SPEED:
                    self._caught = True
                    self._catch_speed = speed
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = "catch the rolling ball"
            ep_meta["catch_layout"] = dict(self.layout)
        return ep_meta

    def _robot_frame(self, p):
        base_pos, _, forward, right, top = self.frame
        d = p - base_pos
        return float(d @ forward), float(d @ right), float(p[2] - top)

    def _ball_held_and_lifted(self) -> bool:
        side = self.rolling_side
        held = any(s == side for s in self._held.values())
        return held and self._robot_frame(self.ball_pos(side))[2] > BALL_RADIUS + LIFT_HEIGHT

    def _check_success(self):
        return self._caught and self._ball_held_and_lifted()

    def phase_signals(self):
        return dict(released=self._released, caught=self._caught, lifted=self._caught and self._ball_held_and_lifted())

    def diagnostic_signals(self):
        return dict(rolling_left=self.rolling_side == "left",
                    held_resting_ball=any(s is not None and s != self.rolling_side for s in self._held.values()))


ENV_NAMES = make_level_classes(CatchRollingBall, __name__)
