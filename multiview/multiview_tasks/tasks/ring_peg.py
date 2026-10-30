"""Task 3: Ring on Peg (precise alignment under self-occlusion).

The robot starts holding a flat ring (washer) in its right hand by a short T handle
(a post on the rim topped by a front-back crossbar). A peg stands on a raised
platform in front of the right arm. The robot carries the ring over the peg, lowers
it so the peg passes through the hole, and opens the hand. The instruction is fixed:
"put the ring on the peg".

While the ring is lowered, the ring and the hand cover the peg top from the head
camera, so the final alignment cannot be seen from egoview (and the single-frame
policy cannot remember it); the wrist-under views look down along the peg from below
the hand and the shoulder views see the ring's height and offset from the side.
Nothing here depends on color.

Knob: the radial clearance between the hole and the peg. The arm motion is the same
at every level; a large clearance tolerates an open-loop drop, a small one needs the
last millimetres corrected while the head view is occluded. The peg position is
randomized per episode from the env rng, identically at every level for a seed.

Holding: the episode starts with the fingers closed around the crossbar and the ring
welded to the palm (scripted grasping of the handle with GR1's hand was unreliable,
and grasping is not what this task measures). The weld releases, for the rest of the
episode, once the right hand opens.
"""

import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.tabletop.tabletop import FixtureType, Tabletop
from robosuite.utils.mjcf_utils import array_to_string
from scipy.spatial.transform import Rotation

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.variant import VariantMixin, make_level_classes

PREFIX = "mvring_"
LANGUAGE = "put the ring on the peg"

# A raised work surface on the counter, in front of the robot's right arm. The counter
# is below GR1's pelvis; without the platform the right hand cannot lower the ring onto
# the peg without bending the waist far forward.
PLATFORM_HEIGHT = 0.22
PLATFORM_FORWARD = (0.20, 0.48)  # extent ahead of the robot base (m)
PLATFORM_RIGHT = (-0.04, 0.44)  # extent to its right (m)

PEG_RADIUS = 0.012
PEG_TIP_LENGTH = 0.025
PEG_HEIGHT = 0.15  # tall enough that the hand can lower the ring deep onto it
PEG_FORWARD = 0.33  # peg axis ahead of the robot base (m); with PEG_RIGHT, where the
PEG_RIGHT = 0.21  # right hand can still lower the ring deep onto the peg
PEG_POS_NOISE = 0.025

RING_WIDTH = 0.014  # radial width of the washer (m)
RING_THICKNESS = 0.008
RING_SEGMENTS = 24
RING_MASS = 0.06
# T handle: a post rising from the rim (ring +x), topped by a crossbar along ring y.
POST_RADIUS = 0.008
POST_HEIGHT = 0.04  # crossbar axis above the ring's mid-plane (m)
BAR_RADIUS = 0.008
BAR_HALF_LENGTH = 0.05

HAND_BODY = "gripper0_right_r_palm"
GRIP_SITE = "gripper0_right_grip_site"
# Crossbar center relative to the right grip site in GR1's rest hand pose, along the
# robot's (forward, right, up) axes (m): where the closed fingers wrap a front-back
# bar (palm faces left, fingers up and curling left and down).
BAR_IN_HAND_FRU = np.array([0.06, -0.036, 0.065])
# Right hand joint targets while holding: [pinky, ring, middle, index, thumb bend,
# thumb rotation] (rad); expanded to the 11 hand joints like FourierHand.format_action.
# Only lightly closed: a tight fist pushes on the crossbar and tilts the welded ring.
HOLD_HAND = np.array([1.0, 1.0, 1.0, 1.0, 0.7, 1.0])
# Weld solref (time constant, damping ratio): stiffer than MuJoCo's default 0.02 s so
# the held ring stays level (~4 deg tilt instead of ~20); 0.004 s = 2 physics steps.
HOLD_SOLREF = "0.004 1"
HAND_JOINT_SOURCE = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
RELEASE_FINGER_ANGLE = 0.7  # mean finger flexion (rad) below which the hand has let go

# The peg counts as through the ring when its axis crosses the ring's plane inside the
# hole, at least this far below the peg top (with 3 mm clearance a slightly tilted ring
# can stop short of the base). A ring lying or leaning beside the peg fails the test
# however close its center is.
INSERT_DEPTH = 0.005


class RingOnPeg(VariantMixin, MultiViewCameraMixin, Tabletop):
    TASK_NAME = "RingOnPeg"
    VALID_LAYOUTS = [0]
    # Radial clearance between the ring hole and the peg (m).
    LEVELS = {
        0: dict(clearance=0.015),
        1: dict(clearance=0.010),
        2: dict(clearance=0.006),
        3: dict(clearance=0.003),
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

    @property
    def inner_radius(self) -> float:
        return PEG_RADIUS + self.variant_params["clearance"]

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        if self.model.worldbody.find(f"./body[@name='{PREFIX}peg']") is None:
            self._add_scene()

    def _sample_noise(self):
        if "ring_peg_noise" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["ring_peg_noise"])
        peg = self.rng.uniform(-PEG_POS_NOISE, PEG_POS_NOISE, size=2)
        return dict(peg_forward=float(peg[0]), peg_right=float(peg[1]))

    def _add_scene(self):
        self.noise = n = self._sample_noise()
        base_pos, base_ori = self.compute_robot_base_placement_pose()
        yaw = base_ori[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
        top = self.counter.get_ext_sites(relative=False)[3][2]

        f0, f1 = PLATFORM_FORWARD
        r0, r1 = PLATFORM_RIGHT
        center = base_pos + (f0 + f1) / 2 * forward + (r0 + r1) / 2 * right
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        platform = ET.Element(
            "body", name=f"{PREFIX}platform",
            pos=array_to_string([center[0], center[1], top + PLATFORM_HEIGHT / 2]),
            quat=array_to_string([qw, qx, qy, qz]),
        )
        ET.SubElement(
            platform, "geom", name=f"{PREFIX}platform_geom", type="box",
            size=array_to_string([(f1 - f0) / 2, (r1 - r0) / 2, PLATFORM_HEIGHT / 2]),
            rgba="0.62 0.52 0.40 1", group="1",
        )
        self.model.worldbody.append(platform)
        top += PLATFORM_HEIGHT

        peg_xy = base_pos + (PEG_FORWARD + n["peg_forward"]) * forward + (PEG_RIGHT + n["peg_right"]) * right
        peg = ET.Element("body", name=f"{PREFIX}peg", pos=array_to_string([peg_xy[0], peg_xy[1], top]))
        # The peg is two stacked cylinders so the top PEG_TIP_LENGTH can be told apart in
        # segmentation renders (visibility analysis); physically it is one peg.
        shaft = PEG_HEIGHT - PEG_TIP_LENGTH
        for name, half, center in (("peg_geom", shaft / 2, shaft / 2), ("peg_tip", PEG_TIP_LENGTH / 2, shaft + PEG_TIP_LENGTH / 2)):
            ET.SubElement(
                peg, "geom", name=f"{PREFIX}{name}", type="cylinder",
                size=f"{PEG_RADIUS} {half}", pos=f"0 0 {center}",
                rgba="0.35 0.38 0.42 1", group="1", friction="0.15 0.005 0.0001",
            )
        ET.SubElement(
            peg, "geom", name=f"{PREFIX}peg_base", type="cylinder", size="0.035 0.004",
            pos="0 0 0.004", rgba="0.35 0.38 0.42 1", group="1",
        )
        self.model.worldbody.append(peg)

        # The ring is placed in the hand at reset (_hold_ring); this pose only needs to
        # be free of contacts for model compilation.
        ring = ET.Element("body", name=f"{PREFIX}ring", pos=array_to_string([peg_xy[0], peg_xy[1], top + 0.3]))
        ET.SubElement(ring, "freejoint", name=f"{PREFIX}ring_joint")
        # Visual-group geoms are excluded from inertia, so the free body gets its own.
        ET.SubElement(ring, "inertial", pos="0 0 0", mass=str(RING_MASS), diaginertia="6e-5 6e-5 1e-4")
        r_in = self.inner_radius
        r_mid = r_in + RING_WIDTH / 2
        half_tangent = np.pi * r_mid / RING_SEGMENTS * 1.08  # slight overlap between segments
        for i in range(RING_SEGMENTS):
            a = 2 * np.pi * i / RING_SEGMENTS
            ET.SubElement(
                ring, "geom", name=f"{PREFIX}ring_seg{i}", type="box",
                size=array_to_string([RING_WIDTH / 2, half_tangent, RING_THICKNESS / 2]),
                pos=array_to_string([r_mid * np.cos(a), r_mid * np.sin(a), 0.0]),
                euler=f"0 0 {a}", rgba="0.85 0.45 0.12 1", group="1",
                friction="0.15 0.005 0.0001",
            )
        self.handle_x = r_in + RING_WIDTH - POST_RADIUS
        ET.SubElement(
            ring, "geom", name=f"{PREFIX}post", type="cylinder",
            size=f"{POST_RADIUS} {POST_HEIGHT / 2}",
            pos=array_to_string([self.handle_x, 0.0, POST_HEIGHT / 2]),
            rgba="0.85 0.45 0.12 1", group="1",
        )
        ET.SubElement(
            ring, "geom", name=f"{PREFIX}handle", type="cylinder",
            size=f"{BAR_RADIUS} {BAR_HALF_LENGTH}",
            pos=array_to_string([self.handle_x, 0.0, POST_HEIGHT]), euler="1.5708 0 0",
            rgba="0.85 0.45 0.12 1", group="1", friction="1.5 0.01 0.001",
        )
        self.model.worldbody.append(ring)
        ET.SubElement(
            self.model.equality, "weld", name=f"{PREFIX}hold",
            body1=HAND_BODY, body2=f"{PREFIX}ring", active="false", solref=HOLD_SOLREF,
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._hold_ring()

    def _hold_ring(self):
        """Close the right hand and put the ring's crossbar in it, welded to the palm."""
        sim, data = self.sim, self.sim.data
        model = sim.model._model
        robot = self.robots[0]
        data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = HOLD_HAND[HAND_JOINT_SOURCE]
        sim.forward()

        base_rot = data.body_xmat[sim.model.body_name2id("robot0_base")].reshape(3, 3)
        forward, right, up = base_rot[:, 0], -base_rot[:, 1], np.array([0.0, 0.0, 1.0])
        site = sim.model.site_name2id(GRIP_SITE)
        bar = data.site_xpos[site] + BAR_IN_HAND_FRU @ np.stack([forward, right, up])
        # Ring frame: x toward the robot's right (post side), z up, crossbar along y.
        ring_rot = np.stack([right, np.cross(up, right), up], axis=1)
        ring_pos = bar - self.handle_x * right - POST_HEIGHT * up
        qx, qy, qz, qw = Rotation.from_matrix(ring_rot).as_quat()

        joint = model.joint(f"{PREFIX}ring_joint")
        qadr, vadr = model.jnt_qposadr[joint.id], model.jnt_dofadr[joint.id]
        data.qpos[qadr : qadr + 7] = np.concatenate([ring_pos, [qw, qx, qy, qz]])
        data.qvel[vadr : vadr + 6] = 0.0
        sim.forward()

        # Weld at the current relative pose: eq_data = anchor(3), relpos(3), relquat(4), torquescale.
        eq = model.equality(f"{PREFIX}hold").id
        hand = sim.model.body_name2id(HAND_BODY)
        hand_rot = data.body_xmat[hand].reshape(3, 3)
        rel = Rotation.from_matrix(hand_rot.T @ ring_rot).as_quat()  # x, y, z, w
        model.eq_data[eq, 0:3] = 0.0
        model.eq_data[eq, 3:6] = hand_rot.T @ (ring_pos - data.body_xpos[hand])
        model.eq_data[eq, 6:10] = [rel[3], rel[0], rel[1], rel[2]]
        model.eq_data[eq, 10] = 1.0
        data._data.eq_active[eq] = 1
        self._hold_eq = eq
        self._released = False
        sim.forward()

    def _finger_flexion(self) -> float:
        q = self.sim.data.qpos[self.robots[0]._ref_gripper_joint_pos_indexes["right"]]
        return float(np.mean(q[:8]))  # pinky, ring, middle and index joints

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        if not self._released and self._finger_flexion() < RELEASE_FINGER_ANGLE:
            self.sim.data._data.eq_active[self._hold_eq] = 0
            self._released = True
        return reward, done, info

    # ------------------------------------------------------------------
    # Task logic
    # ------------------------------------------------------------------
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = LANGUAGE
        if hasattr(self, "noise"):
            ep_meta["ring_peg_noise"] = dict(self.noise)
        return ep_meta

    def peg_top(self) -> np.ndarray:
        """Center of the peg's top face (world)."""
        pos = self.sim.data.body_xpos[self.sim.model.body_name2id(f"{PREFIX}peg")].copy()
        pos[2] += PEG_HEIGHT
        return pos

    def ring_pose(self):
        body = self.sim.model.body_name2id(f"{PREFIX}ring")
        return self.sim.data.body_xpos[body].copy(), self.sim.data.body_xmat[body].reshape(3, 3).copy()

    def handle_center(self) -> np.ndarray:
        """Center of the crossbar (world)."""
        geom = self.sim.model.geom_name2id(f"{PREFIX}handle")
        return self.sim.data.geom_xpos[geom].copy()

    def _ring_offset(self):
        """Horizontal distance of the ring center from the peg axis, and ring height
        relative to the peg top (both in metres)."""
        center, _ = self.ring_pose()
        top = self.peg_top()
        return float(np.linalg.norm(center[:2] - top[:2])), float(center[2] - top[2])

    def _is_inserted(self) -> bool:
        """True when the peg axis passes through the ring's hole below the peg top."""
        center, rot = self.ring_pose()
        normal = rot[:, 2]
        base = self.peg_top() - np.array([0.0, 0.0, PEG_HEIGHT])
        if abs(normal[2]) < 1e-3:  # ring plane parallel to the peg axis
            return False
        s = np.dot(normal, center - base) / normal[2]  # height of the crossing above the peg base
        crossing = base + np.array([0.0, 0.0, s])
        inside_hole = np.linalg.norm(crossing - center) < self.inner_radius
        return bool(inside_hole and 0.0 < s < PEG_HEIGHT - INSERT_DEPTH)

    def _check_success(self):
        return self._released and self._is_inserted()

    def phase_signals(self):
        horizontal, height = self._ring_offset()
        return dict(
            above_peg=horizontal < self.inner_radius and height > 0.0,
            released=self._released,
            inserted=self._is_inserted(),
        )


ENV_NAMES = make_level_classes(RingOnPeg, __name__)
