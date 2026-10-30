"""Task 7 (prototype): Press the Button that Lights Up (field of view, reaction).

An upright panel stands on the work surface in front of the robot, as two wings (left
and right of the midline), each turned PANEL_YAW_DEG toward its side so that hand's index
finger points straight into it. Each wing carries six spring-loaded buttons in three
rows of two. All buttons are grey. At a random moment one of the twelve lights up (turns
amber) and stays lit for LIT_STEPS; the robot has to press that button while it is lit,
without pressing any other button first: "press the button that lights up".

What has to be perceived: which button lights up, and when. The head camera looks down
at the work surface, so the upper rows of the panel are above the top of its image;
cameras that look higher see them. Pressing a wrong button ends the attempt, so a
policy that cannot see the lit button can only guess one. The waist is locked.

Knob: the heights of the three rows. At L0 every row is low enough for the head camera;
at higher levels the upper rows move up, out of its image (and the top row out of
cameras that look only slightly higher).
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.variant import VariantMixin, make_level_classes

PREFIX = "mvlit_"
SIDES = {"left": -1.0, "right": 1.0}  # sign of the robot-frame "right" coordinate

# Raised work surface (robot frame: forward, right; metres from the robot base), as in
# the other multi-view tasks.
PLATFORM_HEIGHT = 0.22
PLATFORM_FORWARD = (0.18, 0.62)
PLATFORM_HALF_WIDTH = 0.50

# Panel wings: center of each wing's face (forward, |right|); the face normal points
# back toward the robot, turned PANEL_YAW_DEG toward that wing's side.
PANEL_CENTER = (0.40, 0.18)
PANEL_YAW_DEG = 25.0
PANEL_HALF_WIDTH = 0.14
PANEL_HEIGHT = 0.70
PANEL_THICKNESS = 0.02
PANEL_RGBA = "0.82 0.82 0.80 1"
COLUMN_OFFSETS = (-0.065, 0.065)  # along the face, from the wing center (inner, outer)

BUTTON_RADIUS = 0.03  # the extended arm lags ~1.5 cm under load; a smaller face gets its rim hit and the finger buckles
BUTTON_HALF_LENGTH = 0.012  # buttons stand 2 cm off the panel
BUTTON_TRAVEL = 0.012
PRESS_THRESHOLD = 0.004  # joint displacement counted as pressed (m), as in ColorButtonBox
BUTTON_STIFFNESS = 40.0  # N/m (the index finger is weak; see ColorButtonBox)
BUTTON_DAMPING = 1.0
UNLIT_RGBA = (0.55, 0.56, 0.58, 1.0)
LIT_RGBA = (1.0, 0.72, 0.08, 1.0)

LIGHT_STEPS = (10, 50)  # the light comes on at a random step in this range (20 Hz)
LIT_STEPS = 100  # and stays on this long (5 s) unless the button is pressed


class PressLitButton(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "PressLitButton"
    VALID_LAYOUTS = [0]
    LOCK_WAIST = True
    # rows: button-center heights above the work surface, bottom to top (m).
    LEVELS = {
        0: dict(rows=(0.10, 0.20, 0.30)),
        1: dict(rows=(0.14, 0.28, 0.42)),
        2: dict(rows=(0.18, 0.40, 0.50)),
        3: dict(rows=(0.20, 0.42, 0.58)),
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
        if self.model.worldbody.find(f"./body[@name='{PREFIX}wing_right']") is None:
            self._add_scene()

    def _sample_layout(self):
        if "lit_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["lit_layout"])
        return dict(
            button=str(self.rng.choice(self.button_names())),
            light_step=int(self.rng.integers(LIGHT_STEPS[0], LIGHT_STEPS[1] + 1)),
        )

    @staticmethod
    def button_names():
        """<side>_r<row>_c<column>; row 0 is the bottom, column 0 the inner one."""
        return [f"{side}_r{row}_c{col}" for side in SIDES for row in range(3) for col in range(2)]

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

    def _robot_frame(self, p):
        base_pos, _, forward, right, top = self.frame
        d = p - base_pos
        return float(d @ forward), float(d @ right), float(p[2] - top)

    def _quat(self, turn_deg=0.0):
        """wxyz of the robot's yaw turned `turn_deg` toward the robot's right."""
        _, yaw, _, _, _ = self.frame
        qx, qy, qz, qw = Rotation.from_euler("z", yaw - np.deg2rad(turn_deg)).as_quat()
        return [qw, qx, qy, qz]

    def _add_scene(self):
        self.layout = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        f0, f1 = PLATFORM_FORWARD
        plat = ET.SubElement(wb, "body", name=f"{PREFIX}platform",
                             pos=array_to_string(self._world((f0 + f1) / 2, 0.0, -PLATFORM_HEIGHT / 2)),
                             quat=array_to_string(self._quat()))
        ET.SubElement(plat, "geom", name=f"{PREFIX}platform_geom", type="box",
                      size=array_to_string([(f1 - f0) / 2, PLATFORM_HALF_WIDTH, PLATFORM_HEIGHT / 2]),
                      rgba="0.62 0.52 0.40 1", group="1")
        rows = self.variant_params["rows"]
        for side, sgn in SIDES.items():
            # Wing body frame: x = face normal (toward the robot), y = along the face
            # (toward the robot's right), z = up; origin at the face center on the work surface.
            turn = -sgn * PANEL_YAW_DEG  # the normal (-forward) turned toward this side
            center = self._world(PANEL_CENTER[0], sgn * PANEL_CENTER[1], 0.0)
            wing = ET.SubElement(wb, "body", name=f"{PREFIX}wing_{side}", pos=array_to_string(center),
                                 quat=array_to_string(self._quat(turn_deg=180.0 + turn)))
            ET.SubElement(wing, "geom", name=f"{PREFIX}wing_{side}_geom", type="box",
                          pos=array_to_string([-PANEL_THICKNESS / 2, 0.0, PANEL_HEIGHT / 2]),
                          size=array_to_string([PANEL_THICKNESS / 2, PANEL_HALF_WIDTH, PANEL_HEIGHT / 2]),
                          rgba=PANEL_RGBA, group="1")
            for row, z in enumerate(rows):
                for col, off in enumerate(COLUMN_OFFSETS):
                    name = f"{side}_r{row}_c{col}"
                    y = sgn * off  # negative offsets (column 0) toward the midline
                    button = ET.SubElement(wing, "body", name=f"{PREFIX}button_{name}", pos=array_to_string([0.0, y, z]))
                    ET.SubElement(button, "inertial", pos="0 0 0", mass="0.02", diaginertia="1e-5 1e-5 1e-5")
                    ET.SubElement(button, "joint", name=f"{PREFIX}button_{name}_joint", type="slide", axis="-1 0 0",
                                  range=f"0 {BUTTON_TRAVEL}", limited="true", stiffness=str(BUTTON_STIFFNESS),
                                  damping=str(BUTTON_DAMPING))
                    ET.SubElement(button, "geom", name=f"{PREFIX}button_{name}_geom", type="cylinder",
                                  size=f"{BUTTON_RADIUS} {BUTTON_HALF_LENGTH}", pos=f"{BUTTON_HALF_LENGTH * 0.75} 0 0",
                                  euler="0 1.5708 0", rgba=array_to_string(UNLIT_RGBA), group="1")

    # ------------------------------------------------------------------
    # Episode logic
    # ------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        model = self.sim.model._model
        self._geoms = {n: model.geom(f"{PREFIX}button_{n}_geom").id for n in self.button_names()}
        for g in self._geoms.values():
            model.geom_rgba[g] = UNLIT_RGBA
        self._step = 0
        self._lit = False
        self._lit_since = None
        self._pressed = {n: False for n in self.button_names()}
        self._press_order = []
        self._success = False

    @property
    def target(self) -> str:
        return self.layout["button"]

    def button_depth(self, name) -> float:
        return float(self.sim.data.get_joint_qpos(f"{PREFIX}button_{name}_joint"))

    def button_face_center(self, name) -> np.ndarray:
        """Center of a button's front face (world)."""
        model, data = self.sim.model, self.sim.data
        body = model.body_name2id(f"{PREFIX}button_{name}")
        normal = data.body_xmat[body].reshape(3, 3)[:, 0]
        return data.body_xpos[body] + 1.75 * BUTTON_HALF_LENGTH * normal

    def button_normal(self, name) -> np.ndarray:
        body = self.sim.model.body_name2id(f"{PREFIX}button_{name}")
        return self.sim.data.body_xmat[body].reshape(3, 3)[:, 0].copy()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        model = self.sim.model._model
        self._step += 1
        if not self._lit and self._lit_since is None and self._step >= self.layout["light_step"]:
            self._lit, self._lit_since = True, self._step
            model.geom_rgba[self._geoms[self.target]] = LIT_RGBA
        for n in self.button_names():
            if not self._pressed[n] and self.button_depth(n) > PRESS_THRESHOLD:
                self._pressed[n] = True
                self._press_order.append(n)
                if n == self.target and self._lit and len(self._press_order) == 1:
                    self._success = True
        if self._lit and (self._pressed[self.target] or self._step - self._lit_since >= LIT_STEPS):
            self._lit = False
            model.geom_rgba[self._geoms[self.target]] = UNLIT_RGBA
        return reward, done, info

    def _check_success(self):
        return self._success

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = "press the button that lights up"
            ep_meta["lit_layout"] = dict(self.layout)
        return ep_meta

    def phase_signals(self):
        return dict(lit=self._lit_since is not None, pressed_lit=self._success,
                    pressed_wrong=any(v for n, v in self._pressed.items() if n != self.target))

    def diagnostic_signals(self):
        side, row, col = self.target.split("_")
        signals = {f"target_row{r}": row == f"r{r}" for r in range(3)}
        signals["target_left"] = side == "left"
        signals["pressed_before_light"] = bool(self._press_order) and self._press_order[0] == self.target and not self._success
        return signals


ENV_NAMES = make_level_classes(PressLitButton, __name__)
