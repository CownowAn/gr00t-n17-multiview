"""Push the Drawer with the Named Handle Shut (the ColorButtonBoxV3 family, with drawers).

The ColorButtonBoxV3 scene with its three buttons replaced by three small drawers, all
pulled open. The drawer fronts are plain wood; each carries a round knob of its own color
(red, green, blue; random left-to-right order). The instruction names one: "push the drawer
with the <color> knob closed". The hand on the side the drawer face turns to pushes that
drawer shut, on the front below the knob.

Visibility: as in ColorButtonBoxV3, a hood over the drawer face (here reaching past the
open fronts and their knobs) hides the knobs from the head camera; which way the face
turns is still visible from the hood and box pose, so only the color order is hidden.
Cameras: those of ColorButtonBoxV3 (wrist-under, wrist-inner, side); profiles cam3 (the
wrist-under pair) and cam5w (wrist-under and side pairs).

Drawers: a slide joint per drawer (0 = open, DRAWER_TRAVEL = shut) with a little dry
friction, so a drawer stays where it is left and a brushing hand moves it only slightly.

Success: the target drawer is pushed in by PUSH_THRESHOLD (most of the way shut) and no
other drawer has been pushed in by more than WRONG_THRESHOLD.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.tasks.color_button import (
    BOX_DISTANCE, BOX_LEFT_OFFSET, BOX_POS_NOISE, BOX_PREFIX, COLORS, FACE_YAW_DEG,
    FACE_YAW_NOISE_DEG, HOOD_THICKNESS, ColorButtonBoxV3,
)
from multiview_tasks.variant import make_level_classes

DRAWER_TRAVEL = 0.05  # open drawer front sticks out this far past the box face (m)
FRONT_HALF = (0.008, 0.045, 0.035)  # drawer front panel half sizes (box frame x, y, z)
KNOB_RADIUS = 0.011  # round knob, on a short stem
KNOB_STEM = 0.008  # stem length from the front's outward face to the knob's back
KNOB_HEIGHT = 0.017  # knob center above the front's center
PUSH_HEIGHT = -0.008  # scripted push point below the front's center (under the knob)
TRAY_HALF = (0.06, 0.04, 0.022)  # drawer body behind the front (visual, no collision)
WOOD = "0.72 0.56 0.38 1"
WOOD_SIDE = "0.64 0.49 0.33 1"
DRAWER_FRICTIONLOSS = 0.3  # N
DRAWER_DAMPING = 1.0  # N s/m
PUSH_THRESHOLD = 0.04  # pushed in this far counts as shut (m)
WRONG_THRESHOLD = 0.01  # another drawer pushed in this far counts as a wrong push (m)


class PushDrawerBoxV3(ColorButtonBoxV3):
    TASK_NAME = "PushDrawerBoxV3"
    # Hood past the box face: the open fronts and their knobs, plus the 4 cm that hides
    # ColorButtonBoxV3's buttons.
    LEVELS = {3: dict(hood=DRAWER_TRAVEL + 2 * FRONT_HALF[0] + KNOB_STEM + 2 * KNOB_RADIUS + 0.04)}
    PRESS_THRESHOLD = PUSH_THRESHOLD
    FACE_PROTRUSION = DRAWER_TRAVEL + 2 * FRONT_HALF[0]  # open front face past the box face
    BUTTON_RADIUS = FRONT_HALF[1]  # metric_signals: "within the front" sideways
    # Scripted policy: push a few mm past flush; pre-push point in front of the hood edge.
    TRAJECTORY_FIXED = dict(pre_clearance=0.05, press_depth=DRAWER_TRAVEL + 0.006, press_speed=0.003,
                            settle=10, hold=15)

    def _add_button_box(self):
        if "button_colors" in self._ep_meta:  # replaying a recorded episode
            order = list(self._ep_meta["button_colors"])
            self.target_color = self._ep_meta["target_color"]
            self.box_noise = dict(self._ep_meta["box_noise"])
        else:
            order = [str(c) for c in self.rng.permutation(list(COLORS))]
            self.target_color = str(self.rng.choice(order))
            forward_offset, right_offset = self.rng.uniform(-BOX_POS_NOISE, BOX_POS_NOISE, size=2)
            self.box_noise = dict(
                forward=float(forward_offset),
                right=float(right_offset),
                yaw_deg=float(self.rng.uniform(-FACE_YAW_NOISE_DEG, FACE_YAW_NOISE_DEG)),
                side=str(self.rng.choice(["left", "right"])),
            )
        self.button_colors = order
        self.face_side = self.box_noise.get("side", "right")
        side_sign = 1.0 if self.face_side == "right" else -1.0

        base_pos, base_ori = self.compute_robot_base_placement_pose()
        yaw = base_ori[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        counter_top = self.counter.get_ext_sites(relative=False)[3][2]
        noise = self.box_noise
        center = (base_pos + (BOX_DISTANCE + noise["forward"]) * forward
                  + (noise["right"] - side_sign * BOX_LEFT_OFFSET) * right)
        center[2] = counter_top + self.BOX_HALF_SIZE[2]
        face_yaw = np.deg2rad(FACE_YAW_DEG + noise["yaw_deg"])
        normal = -np.cos(face_yaw) * forward + side_sign * np.sin(face_yaw) * right
        box_yaw = np.arctan2(normal[1], normal[0])
        qx, qy, qz, qw = Rotation.from_euler("z", box_yaw).as_quat()

        body = ET.Element("body", name=f"{BOX_PREFIX}body", pos=array_to_string(center),
                          quat=array_to_string([qw, qx, qy, qz]))
        hx, hy, hz = self.BOX_HALF_SIZE
        ET.SubElement(body, "geom", name=f"{BOX_PREFIX}shell", type="box", size=array_to_string(self.BOX_HALF_SIZE),
                      rgba="0.85 0.85 0.82 1", group="1")
        hood = self.variant_params["hood"]
        ET.SubElement(body, "geom", name=f"{BOX_PREFIX}hood", type="box",
                      size=array_to_string([hx + hood / 2, hy, HOOD_THICKNESS / 2]),
                      pos=array_to_string([hood / 2, 0.0, hz + HOOD_THICKNESS / 2]),
                      rgba="0.55 0.55 0.58 1", group="1")
        fx, fy, fz = FRONT_HALF
        spread = self.BUTTON_SPREAD * hy
        for color, y in zip(order, (-spread, 0.0, spread)):
            # Body origin at the box face; the joint moves it in (-x) by up to DRAWER_TRAVEL.
            drawer = ET.SubElement(body, "body", name=f"{BOX_PREFIX}button_{color}",
                                   pos=array_to_string([hx + DRAWER_TRAVEL, y, 0.0]))
            ET.SubElement(drawer, "inertial", pos="0 0 0", mass="0.2", diaginertia="1e-4 1e-4 1e-4")
            ET.SubElement(drawer, "joint", name=f"{BOX_PREFIX}button_{color}_joint", type="slide", axis="-1 0 0",
                          range=f"0 {DRAWER_TRAVEL}", limited="true", frictionloss=str(DRAWER_FRICTIONLOSS),
                          damping=str(DRAWER_DAMPING))
            ET.SubElement(drawer, "geom", name=f"{BOX_PREFIX}button_{color}_geom", type="box",
                          size=array_to_string(FRONT_HALF), pos=array_to_string([fx, 0, 0]), rgba=WOOD, group="1")
            ET.SubElement(drawer, "geom", name=f"{BOX_PREFIX}button_{color}_tray", type="box",
                          size=array_to_string(TRAY_HALF), pos=array_to_string([-TRAY_HALF[0], 0, -0.004]),
                          rgba=WOOD_SIDE, group="1", contype="0", conaffinity="0")
            # Round knob on a short stem, both in the drawer's color.
            ET.SubElement(drawer, "geom", name=f"{BOX_PREFIX}button_{color}_stem", type="cylinder",
                          size=f"0.004 {KNOB_STEM / 2}", pos=array_to_string([2 * fx + KNOB_STEM / 2, 0, KNOB_HEIGHT]),
                          euler="0 1.5708 0", rgba=array_to_string(COLORS[color]), group="1")
            ET.SubElement(drawer, "geom", name=f"{BOX_PREFIX}button_{color}_handle", type="sphere",
                          size=str(KNOB_RADIUS), pos=array_to_string([2 * fx + KNOB_STEM + KNOB_RADIUS, 0, KNOB_HEIGHT]),
                          rgba=array_to_string(COLORS[color]), group="1")
        self.model.worldbody.append(body)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "target_color"):
            ep_meta["lang"] = f"push the drawer with the {self.target_color} knob closed"
        return ep_meta

    def _update_pressed(self):
        # A drawer counts as pushed once in by WRONG_THRESHOLD (the target: once shut).
        for color in COLORS:
            thr = self.PRESS_THRESHOLD if color == self.target_color else WRONG_THRESHOLD
            if not self._pressed[color] and self._button_depth(color) > thr:
                self._pressed[color] = True
                self._press_order.append(color)

    def target_face_center(self):
        """Push point on the target drawer front's outward face, below the handle (world)."""
        model, data = self.sim.model, self.sim.data
        body = model.body_name2id(f"{BOX_PREFIX}button_{self.target_color}")
        rot = data.body_xmat[body].reshape(3, 3)
        return data.body_xpos[body] + rot @ np.array([2 * FRONT_HALF[0], 0.0, PUSH_HEIGHT])


ENV_NAMES = make_level_classes(PushDrawerBoxV3, __name__)
