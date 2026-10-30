"""Single source of truth for the multi-view camera rig.

Every camera used by the multi-view tasks is defined here: where it is mounted,
its pose relative to the parent body, its field of view, how its render is framed
into the 256x256 observation, and the observation key it is exposed under. Env
construction, data export and (later) the GR00T modality config should all read
from this module.

All poses match the RoboCasa-GR1 "Multiview-15views" camera table (checked by
tests/test_tasks.py::test_camera_poses_match_release), which was defined against
robocasa-gr1-tabletop-tasks 4840e67, robosuite 1.5.1 and mujoco 3.2.6. The mounted
cameras are its wrist-under and shoulder cameras (multiview_cameras.py EXTRA_CAMERAS);
egoview and robotview come with robocasa and robosuite. Observation keys follow that
dataset's stream names.
"""

from dataclasses import dataclass
from typing import Optional

# Image size rendered by robosuite for every camera. The GR1 checkpoint expects
# egoview at 1280x800, and RoboCasaEnv only supports a single size for all cameras.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 800


@dataclass(frozen=True)
class CameraSpec:
    name: str  # MuJoCo camera name (also the robosuite "<name>_image" obs prefix)
    obs_key: str  # GR00T observation key
    parent_body: Optional[str] = None  # None: camera already defined by robocasa
    pos: tuple = ()  # metres, in the parent body frame
    quat: tuple = ()  # wxyz, in the parent body frame
    fovy: float = 0.0  # vertical field of view (degrees)
    # "pad": robocasa process_img (letterbox the full frame to a square).
    # "square": centre 800x800 crop of the 1280x800 render, INTER_AREA resize to 256x256.
    framing: str = "square"


# Defined by robocasa's Tabletop.set_cameras (mounted on robot0_head_pitch, fovy 77).
# The base GR1 checkpoint was trained on this key and framing.
EGOVIEW = CameraSpec(name="egoview", obs_key="video.ego_view_pad_res256_freq20", framing="pad")

# 5 cm below the wrist, aimed at the fingertips (sees the grasp from underneath).
# Left and right are exact sagittal mirrors. GR1's left hand body really is "..._eefd".
LEFT_WRIST_UNDER = CameraSpec(
    name="left_wrist_under",
    obs_key="video.left_wrist_under_view",
    parent_body="robot0_left_hand_eefd",
    pos=(-0.0221, 0.0411, 0.0109),
    quat=(-0.3277, 0.2035, 0.1928, 0.9022),
    fovy=90.0,
)
RIGHT_WRIST_UNDER = CameraSpec(
    name="right_wrist_under",
    obs_key="video.right_wrist_under_view",
    parent_body="robot0_right_hand_eef",
    pos=(-0.0221, -0.0411, 0.0109),
    quat=(0.9022, 0.1928, 0.2035, -0.3277),
    fovy=90.0,
)

# On the shoulder brackets, rigid to the torso (moves with the waist, not the arm).
LEFT_SHOULDER = CameraSpec(
    name="shoulder_left",
    obs_key="video.shoulder_left",
    parent_body="robot0_left_arm_mount",
    pos=(0.1002, 0.0847, 0.1811),
    quat=(0.7051, 0.0539, -0.3468, -0.6162),
    fovy=77.0,
)
RIGHT_SHOULDER = CameraSpec(
    name="shoulder_right",
    obs_key="video.shoulder_right",
    parent_body="robot0_right_arm_mount",
    pos=(0.1002, -0.0843, 0.1813),
    quat=(0.6162, 0.3468, -0.0539, -0.7050),
    fovy=77.0,
)


# Pelvis cameras: on the robot base (below the waist joints, so the waist does not move
# them), a mirrored pair 20 cm ahead, +-22 cm to the side, 25 cm above the base (8 cm
# above the 22 cm work surfaces of the multi-view tasks), both aimed at the midline point
# 42 cm ahead at shelf-block height. Wide enough apart that the reaching right arm never
# covers both (closer in, it passes in front of the right one). A viewpoint the head
# cannot reach by leaning: used by ShelfTakeBlock (cam5p), not part of MOUNTED_CAMERAS.
PELVIS_LEFT = CameraSpec(
    name="pelvis_left",
    obs_key="video.pelvis_left",
    parent_body="robot0_base",
    pos=(0.20, 0.22, 0.25),
    quat=(0.2636, 0.2775, -0.6698, -0.6363),  # xyaxes "-0.707 -0.707 0 -0.036 0.036 0.999"
    fovy=77.0,
)
PELVIS_RIGHT = CameraSpec(
    name="pelvis_right",
    obs_key="video.pelvis_right",
    parent_body="robot0_base",
    pos=(0.20, -0.22, 0.25),
    quat=(0.6363, 0.6698, -0.2775, -0.2636),  # xyaxes "0.707 -0.707 0 -0.036 -0.036 0.999"
    fovy=77.0,
)
PELVIS_CAMERAS = (PELVIS_LEFT, PELVIS_RIGHT)

# Hand-front cameras: under the wrist (4 cm below and 8 cm behind the right grasp point
# when the hand is palm down with the fingers 25 deg left of forward), looking 10 deg
# down and 40 deg left of forward, with a wide 100 deg field of view. Turned this far
# inward, the resting hand's camera sees no block (at 25 deg its view edge caught the shelf).
# They only see what is in front of the reaching hand. Left is the sagittal mirror of right (pos y and camera x negated). Used by
# ShelfTakeBlock (cam3p, cam5p) instead of the wrist-under cameras.
RIGHT_HAND_FRONT = CameraSpec(
    name="right_hand_front",
    obs_key="video.right_hand_front",
    parent_body="robot0_right_hand_eef",
    pos=(-0.0347, 0.0349, -0.0118),
    quat=(-0.5431, 0.0207, -0.4561, 0.7047),
    fovy=100.0,
)
LEFT_HAND_FRONT = CameraSpec(
    name="left_hand_front",
    obs_key="video.left_hand_front",
    parent_body="robot0_left_hand_eefd",
    pos=(-0.0347, -0.0349, -0.0118),
    quat=(0.7047, -0.4561, 0.0207, -0.5431),
    fovy=100.0,
)
HAND_FRONT_CAMERAS = (LEFT_HAND_FRONT, RIGHT_HAND_FRONT)

# ColorButtonBoxV3 rig. Side cameras: on the robot base, a mirrored pair 5 cm ahead, 20 cm
# to the side and 22 cm up (about elbow height beside the torso), each aimed at the mean
# middle-button position of the box face turned to its side; below the hood, so they see
# the colors the head cannot, and behind the reaching arm, so it seldom covers them.
SIDE_LEFT = CameraSpec(
    name="side_left",
    obs_key="video.side_left",
    parent_body="robot0_base",
    pos=(0.05, 0.20, 0.22),
    quat=(-0.4023, -0.2435, 0.4569, 0.755),
    fovy=60.0,
)
SIDE_RIGHT = CameraSpec(
    name="side_right",
    obs_key="video.side_right",
    parent_body="robot0_base",
    pos=(0.05, -0.20, 0.22),
    quat=(0.755, 0.4569, -0.2435, -0.4023),
    fovy=60.0,
)
SIDE_CAMERAS = (SIDE_LEFT, SIDE_RIGHT)
# Wrist-inner cameras: on the hand, 12 cm behind, 3 cm below and 5 cm to the thumb side of
# the pointing fingertip (when it points at the button), aimed at the button. Looking up
# from the resting hand they see no button; they see the face only once the hand is in
# front of it (and stay above the table top). Left is the sagittal mirror of right.
LEFT_WRIST_INNER = CameraSpec(
    name="left_wrist_inner",
    obs_key="video.left_wrist_inner",
    parent_body="robot0_left_hand_eefd",
    pos=(-0.0612, 0.0083, -0.0928),
    quat=(-0.2243, -0.0949, 0.2239, 0.9437),
    fovy=77.0,
)
RIGHT_WRIST_INNER = CameraSpec(
    name="right_wrist_inner",
    obs_key="video.right_wrist_inner",
    parent_body="robot0_right_hand_eef",
    pos=(-0.0612, -0.0083, -0.0928),
    quat=(0.9437, 0.2239, -0.0949, -0.2243),
    fovy=77.0,
)
WRIST_INNER_CAMERAS = (LEFT_WRIST_INNER, RIGHT_WRIST_INNER)
# HoldDoorTakeBlockV3 right-side cameras on robot0_base, near the right shoulder camera (chosen
# by a search over positions within 30 cm of it that the right arm never passes within 7 cm of;
# both aim at the middle of the block row). Low: 20 cm below / 5 cm ahead of the shoulder
# camera, under the reaching arm: sees the target unoccluded through the grasp (0.90 of its
# pixels, shoulder camera 0.31). High: 20 cm above, 15 cm ahead, 5 cm right: sees all three
# blocks from the start (0.95 of the early frames) and the hand all the way to the drop.
DOOR_LOW_RIGHT = CameraSpec(
    name="door_low_right",
    obs_key="video.door_low_right",
    parent_body="robot0_base",
    pos=(0.1099, -0.1442, 0.3644),
    quat=(-0.3319, -0.3504, 0.6359, 0.6023),
    fovy=70.0,
)
DOOR_HIGH_RIGHT = CameraSpec(
    name="door_high_right",
    obs_key="video.door_high_right",
    parent_body="robot0_base",
    pos=(0.2099, -0.1942, 0.7644),
    quat=(-0.4414, -0.1479, 0.2812, 0.8392),
    fovy=70.0,
)
DOOR_CAMERAS = (DOOR_LOW_RIGHT, DOOR_HIGH_RIGHT)
# StopRollingCanTable body cameras on robot0_base at the shoulder cameras' positions, looking
# out over the table: "diag" 30 deg out from forward, 20 deg down, fovy 45 (sees the cans that
# come diagonally from the far corners start); "side" 5 cm further out and back, 90 deg out,
# 20 deg down, fovy 70 (sees the cans that come along the table from its ends start).
ROLL_DIAG_LEFT = CameraSpec(name="roll_diag_left", obs_key="video.roll_diag_left", parent_body="robot0_base",
                            pos=(0.0282, 0.119, 0.575), quat=(0.7094, 0.4967, -0.2868, -0.4096), fovy=45.0)
ROLL_DIAG_RIGHT = CameraSpec(name="roll_diag_right", obs_key="video.roll_diag_right", parent_body="robot0_base",
                             pos=(0.0235, -0.1341, 0.5707), quat=(-0.4096, -0.2868, 0.4967, 0.7094), fovy=45.0)
ROLL_SIDE_LEFT = CameraSpec(name="roll_side_left", obs_key="video.roll_side_left", parent_body="robot0_base",
                            pos=(-0.0218, 0.169, 0.575), quat=(0.8192, 0.5736, 0.0, 0.0), fovy=70.0)
ROLL_SIDE_RIGHT = CameraSpec(name="roll_side_right", obs_key="video.roll_side_right", parent_body="robot0_base",
                             pos=(-0.0265, -0.1841, 0.5707), quat=(0.0, 0.0, 0.5736, 0.8192), fovy=70.0)
ROLL_CAMERAS = (ROLL_DIAG_LEFT, ROLL_DIAG_RIGHT, ROLL_SIDE_LEFT, ROLL_SIDE_RIGHT)

# Defined by robosuite's GR1 robot.xml on robot0_head_pitch: pos (0.13, 0, 0.09),
# xyaxes "0 -1 0 0.9 0 0.6", fovy 50. A narrower head camera than egoview.
ROBOTVIEW = CameraSpec(name="robot0_robotview", obs_key="video.robotview")

# Cameras added to every multi-view scene, regardless of which ones are observed.
MOUNTED_CAMERAS = (LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER)

# Which cameras are rendered and returned as observations. The scene is the same for all.
# The egoview spec yields both ego framings (pad, and bg crop added by GrootRoboCasaEnv).
CAMERA_PROFILES = {
    # cam1 is the base GR1 checkpoint's input; cam3 adds the wrist-under views and
    # cam5 the shoulder views as well.
    "cam1": (EGOVIEW,),
    "cam3": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER),
    "cam5": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER),
    # Everything stored in demonstration datasets.
    "collect": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER),
    # Low-body rig (tasks that also mount PELVIS_CAMERAS and HAND_FRONT_CAMERAS): cam3p uses
    # the hand-front cameras for the wrist pair, cam5p adds the pelvis cameras; collect_p
    # stores every camera.
    "cam3p": (EGOVIEW, LEFT_HAND_FRONT, RIGHT_HAND_FRONT),
    "cam5p": (EGOVIEW, LEFT_HAND_FRONT, RIGHT_HAND_FRONT, PELVIS_LEFT, PELVIS_RIGHT),
    "collect_p": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER,
                  PELVIS_LEFT, PELVIS_RIGHT, LEFT_HAND_FRONT, RIGHT_HAND_FRONT),
    # ColorButtonBoxV3 rig (mounts SIDE_CAMERAS and WRIST_INNER_CAMERAS): cam3b adds the
    # wrist-inner pair, cam5b also the side pair; collect_b stores every camera.
    "cam3b": (EGOVIEW, LEFT_WRIST_INNER, RIGHT_WRIST_INNER),
    "cam5b": (EGOVIEW, LEFT_WRIST_INNER, RIGHT_WRIST_INNER, SIDE_LEFT, SIDE_RIGHT),
    "collect_b": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER,
                  LEFT_WRIST_INNER, RIGHT_WRIST_INNER, SIDE_LEFT, SIDE_RIGHT),
    # HoldDoorTakeBlockV3 rig (mounts DOOR_CAMERAS): cam3 as above (wrist-under pair), cam5d
    # adds the low and high right-side cameras; collect_d stores every camera.
    "cam5d": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, DOOR_LOW_RIGHT, DOOR_HIGH_RIGHT),
    "collect_d": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER,
                  DOOR_LOW_RIGHT, DOOR_HIGH_RIGHT),
    # StopRollingCanTable rig (mounts ROLL_CAMERAS): cam3r adds the diagonal pair, cam5r also
    # the side pair; collect_r stores every camera.
    "cam3r": (EGOVIEW, ROLL_DIAG_LEFT, ROLL_DIAG_RIGHT),
    "cam5r": (EGOVIEW, ROLL_DIAG_LEFT, ROLL_DIAG_RIGHT, ROLL_SIDE_LEFT, ROLL_SIDE_RIGHT),
    "collect_r": (EGOVIEW, ROBOTVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, LEFT_SHOULDER, RIGHT_SHOULDER,
                  ROLL_DIAG_LEFT, ROLL_DIAG_RIGHT, ROLL_SIDE_LEFT, ROLL_SIDE_RIGHT),
    # PushDrawerBoxV3 rig (ColorButtonBoxV3's cameras): cam3 as above (wrist-under pair), cam5w
    # also the side pair; collect_b stores every camera.
    "cam5w": (EGOVIEW, LEFT_WRIST_UNDER, RIGHT_WRIST_UNDER, SIDE_LEFT, SIDE_RIGHT),
}
