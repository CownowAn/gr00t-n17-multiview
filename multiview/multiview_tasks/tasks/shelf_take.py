"""Task 5: Take the Block from the Low Shelf (a viewpoint only body-low cameras have).

A low shelf (closed top, open toward the robot) stands on a raised work surface in front
of the robot. Three colored blocks (red, green, blue) sit deep inside, in three of four
slots. The instruction names one color: "take the <color> block out of the shelf". The
right hand reaches into the shelf, takes that block and puts it down on the work
surface to the right of the shelf.

Visibility (static prototype, full top): the head looks down on the shelf top and never
sees inside, even with the waist free (the side walls reach 18 cm in front of the blocks, so a
head leaning sideways cannot look in at a grazing angle; every waist pose that would let
it see puts the head through the shelf). The pelvis cameras (cameras.PELVIS_CAMERAS, on
the robot base 4 cm above the work surface) look straight into the opening and see every
block from the start; the right wrist-under camera sees inside once the hand is at the
opening. Expected: cam1 (guess) < cam3 (finds the block on the way in) < cam5p (knows it
from the start). The waist is locked (see LOCK_WAIST).

Knob: how many of the four slots the shelf top covers, from the right. Without a top
(L0) the head sees every block; L1 hides slots 2-3, L2 slots 1-3, the full top (L3) all. Walls, slots and arm motion are the same at
every level; the shelf pose and the block layout are randomized per episode from the env
rng, identically at every level for a seed.

Grasping: a closed right hand with a block near its grasp point (a site in the palm frame
where the closed fingertips meet) holds that block by a weld; opening the hand drops it
(GR1's hand cannot reliably pick a block by friction alone; any policy triggers the weld
the same way).
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.cameras import HAND_FRONT_CAMERAS, PELVIS_CAMERAS
from multiview_tasks.variant import VariantMixin, make_level_classes

PREFIX = "mvshelf_"
COLORS = {"red": "0.85 0.15 0.12 1", "green": "0.15 0.65 0.20 1", "blue": "0.15 0.35 0.85 1"}

PLATFORM_HEIGHT = 0.22
PLATFORM_FORWARD = (0.18, 0.62)
PLATFORM_RIGHT = (-0.30, 0.50)

# Shelf (robot frame: forward, right from the base; heights above the work surface).
SHELF_MOUTH = 0.32  # front edge of the side walls, 10 cm before the blocks
SHELF_BACK = 0.56
SHELF_CENTER_R = 0.06
SHELF_WIDTH = 0.40  # interior: 10 cm right of the rightmost slot, so the hand's pinky side clears the wall
SHELF_HEIGHT = 0.14  # interior: room for the palm-down hand
# The shelf stands on a base, so its floor is this far above the work surface: with the
# blocks at work-surface height the right hand could not get low enough into the opening.
SHELF_FLOOR = 0.08
WALL = 0.01
SHELF_POS_NOISE = 0.015
# The shelf top reaches this far in front of the opening (3 cm already hides the blocks from egoview).
ROOF_OVERHANG = 0.04

BLOCK_HALF = 0.016
BLOCK_MASS = 0.05
BLOCK_FORWARD = 0.42
SLOTS_R = (-0.08, 0.0, 0.08, 0.16)  # 8 cm apart; three of four used

RIGHT_PALM = "gripper0_right_r_palm"
RIGHT_GRASP_SITE = f"{PREFIX}right_grasp"
# Where the fully closed fingertips meet, in the right palm frame (measured).
GRASP_POINT_IN_PALM = np.array([0.028, 0.0464, -0.1002])
GRASP_FINGER_ANGLE = 0.35  # fingers closing on a block stop at 0.4-0.6
GRASP_DISTANCE = 0.05  # blocks are 8 cm apart, so the nearest one is the meant one
RELEASE_FINGER_ANGLE = 0.3
HOLD_SOLREF = "0.004 1"


class ShelfTakeBlock(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "ShelfTakeBlock"
    VALID_LAYOUTS = [0]
    # Locked anyway, although the shelf shape alone nearly suffices: with a free waist the
    # best collision-free head pose sees ~17% of a block (static check).
    LOCK_WAIST = True
    # The shelf top (when present) always reaches ROOF_OVERHANG in front of the opening:
    # flush with it, egoview still sees the blocks through the opening (100-700 px of
    # 640x400). Its depth is no good knob (egoview sees every block until the top is within
    # 4 cm of the opening, then none). Instead the levels cover more slots from the right:
    # top_left is the right-coordinate of the top's left edge (None: full width), measured
    # so that egoview sees the blocks in the uncovered slots and none in the covered ones
    # (the head sees under the edge from the left, so it must sit 3-6 cm left of a slot to hide it).
    #   L0 no top: all 4 slots seen; L1: slots 2-3 hidden; L2: slots 1-3; L3: all.
    LEVELS = {
        0: dict(top_front=None),
        1: dict(top_front=SHELF_MOUTH - ROOF_OVERHANG, top_left=0.02),
        2: dict(top_front=SHELF_MOUTH - ROOF_OVERHANG, top_left=-0.03),
        3: dict(top_front=SHELF_MOUTH - ROOF_OVERHANG, top_left=None),
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        super().__init__(*args, **kwargs)

    def set_cameras(self):
        super().set_cameras()
        for spec in PELVIS_CAMERAS + HAND_FRONT_CAMERAS:
            self._cam_configs[spec.name] = dict(
                pos=list(spec.pos), quat=list(spec.quat), parent_body=spec.parent_body,
                camera_attribs=dict(fovy=str(spec.fovy)),
            )

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
        if self.model.worldbody.find(f"./body[@name='{PREFIX}platform']") is None:
            self._add_scene()

    def _sample_layout(self):
        if "shelf_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["shelf_layout"])
        noise = self.rng.uniform(-SHELF_POS_NOISE, SHELF_POS_NOISE, size=2)
        slots = [int(s) for s in self.rng.permutation(len(SLOTS_R))[:3]]
        colors = [str(c) for c in self.rng.permutation(list(COLORS))]
        return dict(shelf_forward=float(noise[0]), shelf_right=float(noise[1]),
                    slots=slots, colors=colors, target=str(self.rng.choice(colors)))

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

    def _box(self, parent, name, center_fru, half, rgba, **geom_kw):
        _, yaw, _, _, _ = self.frame
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        body = ET.SubElement(parent, "body", name=name, pos=array_to_string(self._world(*center_fru)),
                             quat=array_to_string([qw, qx, qy, qz]))
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=array_to_string(half),
                      rgba=rgba, group="1", **geom_kw)
        return body

    def _add_scene(self):
        self.layout = lay = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        f0, f1 = PLATFORM_FORWARD
        r0, r1 = PLATFORM_RIGHT
        self._box(wb, f"{PREFIX}platform", ((f0 + f1) / 2, (r0 + r1) / 2, -PLATFORM_HEIGHT / 2),
                  ((f1 - f0) / 2, (r1 - r0) / 2, PLATFORM_HEIGHT / 2), "0.62 0.52 0.40 1")

        df, dr = lay["shelf_forward"], lay["shelf_right"]
        self.mouth = SHELF_MOUTH + df
        self.shelf_back = SHELF_BACK + df
        self.shelf_r = (SHELF_CENTER_R + dr - SHELF_WIDTH / 2, SHELF_CENTER_R + dr + SHELF_WIDTH / 2)
        F, B, R, W, H, t = self.mouth, self.shelf_back, SHELF_CENTER_R + dr, SHELF_WIDTH, SHELF_HEIGHT, WALL
        Z = SHELF_FLOOR
        grey = "0.80 0.78 0.74 1"
        self._box(wb, f"{PREFIX}base", ((F + B) / 2, R, Z / 2), ((B - F) / 2 + t, W / 2 + t, Z / 2), "0.55 0.55 0.52 1")
        self._box(wb, f"{PREFIX}back", (B + t / 2, R, Z + H / 2), (t / 2, W / 2 + t, H / 2 + t), grey)
        for side, s in (("left", -1), ("right", 1)):
            self._box(wb, f"{PREFIX}{side}", ((F + B) / 2, R + s * (W / 2 + t / 2), Z + H / 2), ((B - F) / 2, t / 2, H / 2), grey)
        top_front = self.variant_params["top_front"]
        if top_front is not None:
            tf = top_front + df
            r0, r1 = R - W / 2 - t, R + W / 2 + t
            left = self.variant_params.get("top_left")
            if left is not None:
                r0 = max(r0, left + dr)
            self._box(wb, f"{PREFIX}top", ((tf + B) / 2, (r0 + r1) / 2, Z + H + t / 2), ((B - tf) / 2, (r1 - r0) / 2, t / 2), grey)

        _, yaw, _, _, _ = self.frame
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        self.block_colors = {}
        for slot, color in zip(lay["slots"], lay["colors"]):
            name = f"{PREFIX}block_{color}"
            b = ET.SubElement(wb, "body", name=name,
                              pos=array_to_string(self._world(BLOCK_FORWARD + df, SLOTS_R[slot] + dr, SHELF_FLOOR + BLOCK_HALF + 0.001)),
                              quat=array_to_string([qw, qx, qy, qz]))
            ET.SubElement(b, "freejoint", name=f"{name}_joint")
            ET.SubElement(b, "inertial", pos="0 0 0", mass=str(BLOCK_MASS), diaginertia="8.5e-6 8.5e-6 8.5e-6")
            ET.SubElement(b, "geom", name=f"{name}_geom", type="box", size=array_to_string([BLOCK_HALF] * 3),
                          rgba=COLORS[color], group="1", friction="1.2 0.01 0.001")
            ET.SubElement(self.model.equality, "weld", name=f"{name}_grasp", body1=RIGHT_PALM, body2=name,
                          active="false", solref=HOLD_SOLREF)
            self.block_colors[color] = slot
        self.target_color = lay["target"]
        palm = self.model.worldbody.find(f".//body[@name='{RIGHT_PALM}']")
        ET.SubElement(palm, "site", name=RIGHT_GRASP_SITE, pos=array_to_string(GRASP_POINT_IN_PALM),
                      size="0.005", group="5")

    # ------------------------------------------------------------------
    # Grasping
    # ------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        self._grasped = None
        self._ever_grasped = set()

    def _finger_flexion(self) -> float:
        q = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes["right"]]
        return float(np.mean(q[:8]))  # pinky, ring, middle and index joints

    def grasp_point(self) -> np.ndarray:
        return self.sim.data.site_xpos[self.sim.model.site_name2id(RIGHT_GRASP_SITE)].copy()

    def block_pos(self, color) -> np.ndarray:
        return self.sim.data.body_xpos[self.sim.model.body_name2id(f"{PREFIX}block_{color}")].copy()

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

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        model, data = self.sim.model._model, self.sim.data._data
        flex = self._finger_flexion()
        if self._grasped is not None and flex < RELEASE_FINGER_ANGLE:
            data.eq_active[model.equality(f"{PREFIX}block_{self._grasped}_grasp").id] = 0
            self._grasped = None
        elif self._grasped is None and flex > GRASP_FINGER_ANGLE:
            point = self.grasp_point()
            dist = {c: np.linalg.norm(self.block_pos(c) - point) for c in self.block_colors}
            color = min(dist, key=dist.get)
            if dist[color] < GRASP_DISTANCE:
                eq = model.equality(f"{PREFIX}block_{color}_grasp").id
                self._set_weld(eq, self.sim.model.body_name2id(RIGHT_PALM),
                               self.sim.model.body_name2id(f"{PREFIX}block_{color}"))
                self._grasped = color
                self._ever_grasped.add(color)
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = f"take the {self.target_color} block out of the shelf"
            ep_meta["shelf_layout"] = dict(self.layout)
            ep_meta["target_color"] = self.target_color
        return ep_meta

    def _robot_frame(self, p):
        """World point -> (forward, right, height above the work surface)."""
        base_pos, _, forward, right, top = self.frame
        d = p - base_pos
        return float(d @ forward), float(d @ right), float(p[2] - top)

    def _inside_shelf(self, color) -> bool:
        f, r, z = self._robot_frame(self.block_pos(color))
        return (self.mouth < f < self.shelf_back and self.shelf_r[0] < r < self.shelf_r[1]
                and SHELF_FLOOR < z < SHELF_FLOOR + SHELF_HEIGHT)

    def _on_work_surface(self, color) -> bool:
        _, _, z = self._robot_frame(self.block_pos(color))
        return z < BLOCK_HALF + 0.01 and self._grasped != color

    def _check_success(self):
        target_out = not self._inside_shelf(self.target_color) and self._on_work_surface(self.target_color)
        return target_out and all(self._inside_shelf(c) for c in self.block_colors if c != self.target_color)

    def phase_signals(self):
        return dict(
            grasped_target=self.target_color in self._ever_grasped,
            target_out=not self._inside_shelf(self.target_color),
            target_on_surface=self._on_work_surface(self.target_color) and not self._inside_shelf(self.target_color),
        )

    def diagnostic_signals(self):
        signals = {f"target_slot{i}": self.block_colors[self.target_color] == i for i in range(len(SLOTS_R))}
        signals["grasped_wrong"] = any(c != self.target_color for c in self._ever_grasped)
        signals["wrong_out"] = any(not self._inside_shelf(c) for c in self.block_colors if c != self.target_color)
        return signals


ENV_NAMES = make_level_classes(ShelfTakeBlock, __name__)
