"""Task 4: Hold the Door, Take the Block (embodiment occlusion by a held door).

A small cabinet stands on a raised work surface just right of the robot's midline.
Its door is hinged on the opening's left edge and has been swung open toward the
robot (95 deg), so the door panel stands upright right in front of the chest. The
left hand holds the door's handle; a spring closes the door if the hand lets go.
Three colored blocks (red, green, blue) sit in three of four slots inside the
cabinet. The instruction names one color: "take the <color> block out of the
cabinet". The right hand reaches in past the door, takes that block and puts it
down on the work surface outside the cabinet.

Visibility: the head camera and the left shoulder camera are left of the door panel,
so the panel hides the cabinet interior from them; the right shoulder camera is right
of the panel and sees every block from the start; the right wrist-under camera sees
the interior only once the hand is inside. Which slot holds the named block is
random, so finding it needs a view of the interior. The waist is locked
(LOCK_WAIST): leaning or twisting would move the head around the panel.

Knob: the door panel's height. A low panel lets the head look over it into the
cabinet; a full-height panel hides the interior. The handle, the left arm pose and
the block layout are the same at every level, and the cabinet pose and block layout
are randomized per episode from the env rng, identically at every level for a seed.

Holding and grasping: the episode starts with the left hand closed on the handle and
welded to the door (scripted grasping of the handle is not what this task measures);
the weld releases for the rest of the episode once the left hand opens. GR1's hand
cannot reliably pick a block by friction alone, so a closed right hand with a block
between its fingers holds that block by a weld (any policy can trigger it the same
way); opening the hand drops it.
"""

import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.cameras import DOOR_CAMERAS
from multiview_tasks.scripted.ik import ArmIK
from multiview_tasks.variant import VariantMixin, make_level_classes

PREFIX = "mvdoor_"
COLORS = {"red": "0.85 0.15 0.12 1", "green": "0.15 0.65 0.20 1", "blue": "0.15 0.35 0.85 1"}

# Raised work surface (robot frame: forward, right; metres from the robot base).
PLATFORM_HEIGHT = 0.22
PLATFORM_FORWARD = (0.18, 0.62)
PLATFORM_RIGHT = (-0.30, 0.50)

# Cabinet: open toward the robot and on its right side (the right forearm reaches in
# diagonally from the right and would rest on a right wall). FRONT is the front
# opening's plane, HINGE_RIGHT its left edge (where the door hinge is). The cabinet
# stands on a base block so its interior is near the head's height and the door panel
# can hide it.
CAB_FRONT = 0.38
CAB_HINGE_RIGHT = 0.06  # door plane ~7 cm right of the head camera, ~5 cm left of the right shoulder camera
CAB_WIDTH = 0.24  # room for the hand beside a block in the right column
CAB_DEPTH = 0.22  # 0.20 left the fingers no room behind the back-left block (wall contact, no grasp)
CAB_HEIGHT = 0.26  # tall enough that the door's top edge is above the head camera
CAB_BASE = 0.20
CAB_WALL = 0.01
CAB_POS_NOISE = 0.015

DOOR_THICKNESS = 0.016
DOOR_OPEN_DEG = 90.0  # 0 = closed; 90 = panel pointing straight at the robot
DOOR_STIFFNESS = 0.3  # Nm/rad toward closed
DOOR_DAMPING = 0.3
DOOR_OPEN_MIN_DEG = 70.0  # success needs the door at least this open
# Handle: a bar along the panel, near its free edge, standing off the face toward the
# robot's left (the side the left hand is on), low enough to fit every panel height.
HANDLE_FROM_EDGE = 0.08  # bar center from the free edge, along the panel (m)
HANDLE_STANDOFF = 0.035  # bar axis from the panel's face (m)
HANDLE_HEIGHT = 0.03  # above the cabinet floor
HANDLE_RADIUS = 0.008
HANDLE_HALF_LENGTH = 0.035

BLOCK_HALF = 0.016
BLOCK_MASS = 0.05
# Slots inside the cabinet: (forward from the opening, right from the hinge edge).
# Gaps of 4-5 cm between blocks leave room for the fingers; the strip just behind the
# hinge stays empty (the right arm cannot reach it past the door). The back row keeps
# 9 cm to the back wall and the left column 11 cm to the left wall, so the open hand
# (fingers forward-left) fits behind the back-left block.
SLOTS = ((0.05, 0.11), (0.05, 0.20), (0.13, 0.11), (0.13, 0.20))

LEFT_PALM = "gripper0_left_l_palm"
LEFT_SITE = "gripper0_left_grip_site"
LEFT_BAR_SITE = f"{PREFIX}left_bar"
RIGHT_GRASP_SITE = f"{PREFIX}right_grasp"  # at GRASP_POINT_IN_PALM
RIGHT_PALM = "gripper0_right_r_palm"
# Handle bar center relative to the left grip site in GR1's rest hand pose, along the
# robot's (forward, right, up) axes: the right-hand crossbar offset of RingOnPeg,
# mirrored (palm faces right, fingers up and curling right and down).
BAR_IN_LEFT_HAND_FRU = np.array([0.06, 0.036, 0.065])
# Hand joint targets while holding [pinky, ring, middle, index, thumb bend, thumb rot].
HOLD_HAND = np.array([1.0, 1.0, 1.0, 1.0, 0.7, 1.0])
HAND_JOINT_SOURCE = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
HOLD_SOLREF = "0.004 1"
RELEASE_FINGER_ANGLE = 0.3  # mean finger flexion (rad) below which a hand has let go
# Right-hand block grasp: the hand is closed past GRASP_FINGER_ANGLE (fingers closing on a
# block, a wall or a neighbor stop at 0.4-0.6) and a block's center is within GRASP_DISTANCE of the
# grasp point, a point fixed in the palm frame where the fully closed fingertips meet
# (fixed, so a half-closed hand does not pick a neighboring block).
GRASP_FINGER_ANGLE = 0.35
GRASP_DISTANCE = 0.05  # blocks are >= 8 cm apart, so the nearest one is the meant one
GRASP_POINT_IN_PALM = np.array([0.028, 0.0464, -0.1002])
FINGERTIP_BODIES = ("R_index_intermediate_link", "R_middle_intermediate_link", "R_thumb_distal_link")


class HoldDoorTakeBlock(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "HoldDoorTakeBlock"
    VALID_LAYOUTS = [0]
    LOCK_WAIST = True
    # Cabinet size, door width and block slots (HoldDoorTakeBlockV2/V3 spread the slots out).
    CAB_DEPTH = CAB_DEPTH
    CAB_WIDTH = CAB_WIDTH
    DOOR_WIDTH = CAB_WIDTH
    SLOTS = SLOTS
    # Door panel height above the cabinet floor (m); the opening is CAB_HEIGHT tall.
    # Comments: block visibility at reset in egoview (visible / unoccluded segmentation pixels,
    # mean over blocks and 30 seeds); shoulder_right sees ~0.95 at every level.
    LEVELS = {
        0: dict(door_height=0.04),  # 0.96
        1: dict(door_height=0.13),  # 0.69
        2: dict(door_height=0.15),  # 0.42
        3: dict(door_height=CAB_HEIGHT + CAB_WALL),  # 0.04 (full height)
    }

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_distractors", False)
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        return []

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}cabinet']") is None:
            self._add_scene()

    def _sample_layout(self):
        if "door_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["door_layout"])
        noise = self.rng.uniform(-CAB_POS_NOISE, CAB_POS_NOISE, size=2)
        slots = [int(s) for s in self.rng.permutation(len(self.SLOTS))[:3]]
        colors = self._sample_items()
        return dict(
            cab_forward=float(noise[0]), cab_right=float(noise[1]),
            slots=slots, colors=colors, target=str(self.rng.choice(colors)),
        )

    def _sample_items(self) -> list:
        """The three items in the cabinet, in slot order (blocks: the colors, permuted)."""
        return [str(c) for c in self.rng.permutation(list(COLORS))]

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

    def _box(self, parent, name, center, half, rgba, **geom_kw):
        _, yaw, _, _, _ = self.frame
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        body = ET.SubElement(parent, "body", name=name, pos=array_to_string(center),
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
        self._box(wb, f"{PREFIX}platform", self._world((f0 + f1) / 2, (r0 + r1) / 2, -PLATFORM_HEIGHT / 2),
                  ((f1 - f0) / 2, (r1 - r0) / 2, PLATFORM_HEIGHT / 2), "0.62 0.52 0.40 1")

        F = CAB_FRONT + lay["cab_forward"]
        HX = CAB_HINGE_RIGHT + lay["cab_right"]
        W, D, H, Z, t = self.CAB_WIDTH, self.CAB_DEPTH, CAB_HEIGHT, CAB_BASE, CAB_WALL
        DW = self.DOOR_WIDTH
        self.cab_front, self.cab_hinge = F, HX
        cab = self._box(wb, f"{PREFIX}cabinet", self._world(F + D / 2, HX + W / 2, Z / 2),
                        (D / 2, W / 2 + t, Z / 2), "0.55 0.55 0.52 1")
        grey = "0.85 0.85 0.82 1"
        # Walls, as geoms of the cabinet body (positions in its frame: x forward, y left).
        for name, pos, half in (
            ("top", (0, 0, Z / 2 + H + t), (D / 2, W / 2 + t, t)),
            ("back", (D / 2 + t, 0, Z / 2 + H / 2), (t, W / 2 + t, H / 2 + t)),
            ("left", (0, W / 2 + t / 2, Z / 2 + H / 2), (D / 2, t / 2, H / 2)),
        ):
            ET.SubElement(cab, "geom", name=f"{PREFIX}cab_{name}", type="box", pos=array_to_string(pos),
                          size=array_to_string(half), rgba=grey, group="1")

        # Door: hinge on the opening's left edge; at angle 0 the panel spans the opening
        # (local -y = robot right); the joint axis points down so positive angles swing
        # the free edge toward the robot.
        dh = self.variant_params["door_height"]
        hinge = self._world(F - DOOR_THICKNESS / 2, HX, Z)
        _, yaw, _, _, _ = self.frame
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        door = ET.SubElement(wb, "body", name=f"{PREFIX}door", pos=array_to_string(hinge),
                             quat=array_to_string([qw, qx, qy, qz]))
        ET.SubElement(door, "joint", name=f"{PREFIX}door_hinge", type="hinge", axis="0 0 -1",
                      range=f"0 {np.deg2rad(110):.4f}", stiffness=str(DOOR_STIFFNESS), springref="0",
                      damping=str(DOOR_DAMPING), armature="0.01")
        ET.SubElement(door, "inertial", pos=f"0 {-DW / 2} {dh / 2}", mass="0.4", diaginertia="0.002 0.002 0.002")
        ET.SubElement(door, "geom", name=f"{PREFIX}door_panel", type="box",
                      pos=array_to_string([0, -DW / 2, dh / 2]),
                      size=array_to_string([DOOR_THICKNESS / 2, DW / 2, dh / 2]),
                      rgba="0.72 0.66 0.56 1", group="1")
        # Handle on the robot-facing face (local -x), near the free edge.
        bar_y = -(DW - HANDLE_FROM_EDGE)
        bar_x = -DOOR_THICKNESS / 2 - HANDLE_STANDOFF
        ET.SubElement(door, "geom", name=f"{PREFIX}handle", type="cylinder",
                      pos=array_to_string([bar_x, bar_y, HANDLE_HEIGHT]), euler="1.5708 0 0",
                      size=f"{HANDLE_RADIUS} {HANDLE_HALF_LENGTH}", rgba="0.25 0.25 0.28 1", group="1",
                      friction="1.5 0.01 0.001")
        for s in (-1, 1):
            ET.SubElement(door, "geom", name=f"{PREFIX}handle_post{s + 1}", type="box",
                          pos=array_to_string([bar_x / 2 - DOOR_THICKNESS / 4, bar_y + s * (HANDLE_HALF_LENGTH - 0.005), HANDLE_HEIGHT]),
                          size=array_to_string([abs(bar_x) / 2 - DOOR_THICKNESS / 4, 0.004, 0.004]),
                          rgba="0.25 0.25 0.28 1", group="1")
        self.handle_local = np.array([bar_x, bar_y, HANDLE_HEIGHT])

        self.block_colors = {}
        for i, (slot, color) in enumerate(zip(lay["slots"], lay["colors"])):
            sf, sr = self.SLOTS[slot]
            body = self._add_item(wb, color, self._world(F + sf, HX + sr, Z), [qw, qx, qy, qz])
            ET.SubElement(self.model.equality, "weld", name=f"{PREFIX}block_{color}_grasp", body1=RIGHT_PALM, body2=body,
                          active="false", solref=HOLD_SOLREF)
            self.block_colors[color] = slot
        self.target_color = lay["target"]
        ET.SubElement(self.model.equality, "weld", name=f"{PREFIX}hold", body1=LEFT_PALM,
                      body2=f"{PREFIX}door", active="false", solref=HOLD_SOLREF)
        # Where the handle bar sits in the closed left hand; placed at reset (_hold_door).
        palm = self.model.worldbody.find(f".//body[@name='{LEFT_PALM}']")
        ET.SubElement(palm, "site", name=LEFT_BAR_SITE, pos="0 0 0", size="0.005", group="5")
        right_palm = self.model.worldbody.find(f".//body[@name='{RIGHT_PALM}']")
        ET.SubElement(right_palm, "site", name=RIGHT_GRASP_SITE, pos=array_to_string(GRASP_POINT_IN_PALM),
                      size="0.005", group="5")

    def _add_item(self, wb, color, floor_point, quat) -> str:
        """Add one item (here a colored block) standing on `floor_point`; returns its body name."""
        name = f"{PREFIX}block_{color}"
        b = ET.SubElement(wb, "body", name=name, pos=array_to_string(floor_point + np.array([0, 0, BLOCK_HALF + 0.001])),
                          quat=array_to_string(quat))
        ET.SubElement(b, "freejoint", name=f"{name}_joint")
        ET.SubElement(b, "inertial", pos="0 0 0", mass=str(BLOCK_MASS), diaginertia="8.5e-6 8.5e-6 8.5e-6")
        ET.SubElement(b, "geom", name=f"{name}_geom", type="box", size=array_to_string([BLOCK_HALF] * 3),
                      rgba=COLORS[color], group="1", friction="1.2 0.01 0.001")
        return name

    def item_half_height(self, color) -> float:
        """Height of an item's center above the surface it rests on (m)."""
        return BLOCK_HALF

    def item_noun(self, color) -> str:
        return f"{color} block"

    # ------------------------------------------------------------------
    # Reset: door open, left hand on the handle
    # ------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        self._grasped = None
        self._ever_grasped = set()
        self._hold_door()

    def _set_door_angle(self, deg):
        model, data = self.sim.model._model, self.sim.data._data
        j = model.joint(f"{PREFIX}door_hinge")
        data.qpos[model.jnt_qposadr[j.id]] = np.deg2rad(deg)
        data.qvel[model.jnt_dofadr[j.id]] = 0.0

    def handle_world(self) -> np.ndarray:
        data = self.sim.data
        body = self.sim.model.body_name2id(f"{PREFIX}door")
        return data.body_xpos[body] + data.body_xmat[body].reshape(3, 3) @ self.handle_local

    def _hold_door(self):
        sim, data = self.sim, self.sim.data
        model = sim.model._model
        robot = self.robots[0]
        self._set_door_angle(DOOR_OPEN_DEG)
        sim.forward()
        base_rot = data.body_xmat[sim.model.body_name2id("robot0_base")].reshape(3, 3)
        fru = np.stack([base_rot[:, 0], -base_rot[:, 1], np.array([0.0, 0.0, 1.0])])
        # Bar site in the palm frame, from the rest hand pose.
        palm = sim.model.body_name2id(LEFT_PALM)
        bar_rest = data.site_xpos[sim.model.site_name2id(LEFT_SITE)] + BAR_IN_LEFT_HAND_FRU @ fru
        bar_site = model.site(LEFT_BAR_SITE).id
        model.site_pos[bar_site] = data.body_xmat[palm].reshape(3, 3).T @ (bar_rest - data.body_xpos[palm])
        sim.forward()
        target = self.handle_world()
        # Left arm to the handle (kinematic, at reset): IK on the sim's own model.
        ik = ArmIK(model, "left", LEFT_BAR_SITE, frame_type="site", orientation_cost=0.1)
        ik.reset(data.qpos)
        rest = ik.frame_pose().rotation()
        goal = mink.SE3.from_rotation_and_translation(rest, target)
        for _ in range(40):
            q = ik.solve(goal, iters=5)
        data.qpos[ik.qpos_idx] = q
        data.qvel[:] = 0.0
        data.qpos[robot._ref_gripper_joint_pos_indexes["left"]] = HOLD_HAND[HAND_JOINT_SOURCE]
        sim.forward()
        self.left_arm_hold = q.copy()
        self.left_reach_error = float(np.linalg.norm(data.site_xpos[bar_site] - target))
        # Weld the door to the palm at the current relative pose.
        eq = model.equality(f"{PREFIX}hold").id
        palm = sim.model.body_name2id(LEFT_PALM)
        door = sim.model.body_name2id(f"{PREFIX}door")
        self._set_weld(eq, palm, door)
        self._hold_eq = eq
        self._door_released = False
        sim.forward()

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

    # ------------------------------------------------------------------
    # Holding and grasping
    # ------------------------------------------------------------------
    def _finger_flexion(self, side) -> float:
        q = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes[side]]
        return float(np.mean(q[:8]))  # pinky, ring, middle and index joints

    def fingertip_center(self) -> np.ndarray:
        data, m = self.sim.data, self.sim.model
        return np.mean([data.body_xpos[m.body_name2id(f"gripper0_right_{b}")] for b in FINGERTIP_BODIES], axis=0)

    def grasp_point(self) -> np.ndarray:
        return self.sim.data.site_xpos[self.sim.model.site_name2id(RIGHT_GRASP_SITE)].copy()

    def _item_body_id(self, color) -> int:
        return self.sim.model.body_name2id(f"{PREFIX}block_{color}")

    def block_pos(self, color) -> np.ndarray:
        """Item center (world)."""
        return self.sim.data.body_xpos[self._item_body_id(color)].copy()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        model, data = self.sim.model._model, self.sim.data._data
        if not self._door_released and self._finger_flexion("left") < RELEASE_FINGER_ANGLE:
            data.eq_active[self._hold_eq] = 0
            self._door_released = True
        flex = self._finger_flexion("right")
        if self._grasped is not None and flex < RELEASE_FINGER_ANGLE:
            data.eq_active[model.equality(f"{PREFIX}block_{self._grasped}_grasp").id] = 0
            self._grasped = None
        elif self._grasped is None and flex > GRASP_FINGER_ANGLE:
            point = self.grasp_point()
            dist = {c: np.linalg.norm(self.block_pos(c) - point) for c in self.block_colors}
            color = min(dist, key=dist.get)
            if dist[color] < GRASP_DISTANCE:
                eq = model.equality(f"{PREFIX}block_{color}_grasp").id
                self._set_weld(eq, self.sim.model.body_name2id(RIGHT_PALM), self._item_body_id(color))
                self._grasped = color
                self._ever_grasped.add(color)
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = f"take the {self.item_noun(self.target_color)} out of the cabinet"
            ep_meta["door_layout"] = dict(self.layout)
            ep_meta["target_color"] = self.target_color
        return ep_meta

    def door_angle_deg(self) -> float:
        return float(np.rad2deg(self.sim.data.get_joint_qpos(f"{PREFIX}door_hinge")))

    def _robot_frame(self, p):
        """World point -> (forward, right, height above the work surface)."""
        base_pos, _, forward, right, top = self.frame
        d = p - base_pos
        return float(d @ forward), float(d @ right), float(p[2] - top)

    def _inside_cabinet(self, color) -> bool:
        f, r, z = self._robot_frame(self.block_pos(color))
        return (self.cab_front < f < self.cab_front + self.CAB_DEPTH and self.cab_hinge < r < self.cab_hinge + self.CAB_WIDTH
                and CAB_BASE < z < CAB_BASE + CAB_HEIGHT)

    def _on_work_surface(self, color) -> bool:
        """Resting on the work surface (not on the cabinet, not held)."""
        _, _, z = self._robot_frame(self.block_pos(color))
        return z < self.item_half_height(color) + 0.01 and self._grasped != color

    def _check_success(self):
        others_inside = all(self._inside_cabinet(c) for c in self.block_colors if c != self.target_color)
        return (self._on_work_surface(self.target_color) and others_inside
                and self.door_angle_deg() > DOOR_OPEN_MIN_DEG)

    def phase_signals(self):
        return dict(
            grasped_target=self.target_color in self._ever_grasped,
            target_out=not self._inside_cabinet(self.target_color),
            target_on_surface=self._on_work_surface(self.target_color),
            door_open=self.door_angle_deg() > DOOR_OPEN_MIN_DEG,
        )

    def episode_record_header(self) -> dict:
        """Per-episode log header (registry GR1_MV_EPISODE_LOG_DIR): target and block slots."""
        return dict(target=self.target_color, block_colors={c: int(v) for c, v in self.block_colors.items()},
                    slots=[list(map(float, x)) for x in self.SLOTS])

    def episode_record_step(self) -> dict:
        """Per-step log: every block's (forward, right, height) in mm, the held block, door angle."""
        return dict(blocks_mm={c: [round(1000 * v) for v in self._robot_frame(self.block_pos(c))] for c in self.block_colors},
                    grasped=self._grasped, door_deg=round(self.door_angle_deg(), 1))

    def diagnostic_signals(self):
        signals = {f"target_slot{i}": self.block_colors[self.target_color] == i for i in range(len(self.SLOTS))}
        signals["grasped_wrong"] = any(c != self.target_color for c in self._ever_grasped)
        signals["wrong_out"] = any(not self._inside_cabinet(c) for c in self.block_colors if c != self.target_color)
        signals["door_released"] = self._door_released
        return signals


class HoldDoorTakeBlockV2(HoldDoorTakeBlock):
    """HoldDoorTakeBlock with the blocks 12 cm apart (V1: 8 cm forward, 9 cm sideways) in a
    6 cm deeper cabinet. With the V1 spacing the palm-down right hand covered a neighboring
    slot while closing on the target, and in more than half the scripted episodes it pushed
    another block by over 2 cm (V2 with the scripted changes below: 8%, scripted success
    56/60 at L3). The door, its handle and the left-arm hold are unchanged.
    """

    TASK_NAME = "HoldDoorTakeBlockV2"
    CAB_DEPTH = 0.28
    SLOTS = ((0.06, 0.10), (0.06, 0.22), (0.18, 0.10), (0.18, 0.22))
    # Scripted fingers turned less far left (deg) for right-column targets.
    FINGER_YAW_RIGHT_COLUMN = (10.0, 30.0)
    # Scripted withdrawal 4 cm further right of the cabinet and a higher carry via point,
    # so the hand passes clear of the front-right block and the held block of the cabinet.
    SCRIPTED_RANGES = dict(
        outside=((-0.045, 0.145, 0.0), (-0.015, 0.175, 0.0)),
        carry_via=((-0.03, 0.0, 0.04), (0.03, 0.06, 0.08)),
    )


class HoldDoorTakeBlockV3(HoldDoorTakeBlockV2):
    """HoldDoorTakeBlockV2 with the three blocks in one row, 20 cm apart, all 7 cm behind the
    opening (V2: a 2x2 grid of slots 12 cm apart, 3 of them filled). Every slot of the row is
    within the right arm's reach (a second row 16-20 cm deeper is not: 1/7 scripted successes
    there), and the hand never brushes a neighbor (V2: 8% of scripted episodes pushed one by
    over 2 cm). The cabinet is wider (56 cm) and shallower (22 cm); the door keeps the V1/V2
    width (24 cm), so the held door, its handle and the left-arm hold are unchanged. The
    cabinet's right side is open, so the opening right of the door's free edge does not change
    what the head camera sees past the open door: nothing inside.
    """

    TASK_NAME = "HoldDoorTakeBlockV3"
    CAB_DEPTH = 0.22
    CAB_WIDTH = 0.56
    SLOTS = ((0.07, 0.10), (0.07, 0.30), (0.07, 0.50))
    # Scripted policy, right slot: reach in and out through the front opening (hand 10-14 cm
    # in front of the block) instead of beside the cabinet's right side, which is beyond the
    # right arm's reach there (IK misses of 12-16 cm).
    SCRIPTED_SLOT_RANGES = {2: dict(outside=((-0.14, 0.02, 0.0), (-0.10, 0.05, 0.0)))}

    def set_cameras(self):
        super().set_cameras()
        for spec in DOOR_CAMERAS:
            self._cam_configs[spec.name] = dict(
                pos=list(spec.pos), quat=list(spec.quat), parent_body=spec.parent_body,
                camera_attribs=dict(fovy=str(spec.fovy)),
            )


ENV_NAMES = make_level_classes(HoldDoorTakeBlock, __name__)
ENV_NAMES_V2 = make_level_classes(HoldDoorTakeBlockV2, __name__)
ENV_NAMES_V3 = make_level_classes(HoldDoorTakeBlockV3, __name__)
