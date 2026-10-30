"""Checks for multi-view task variants. Run with the robocasa_uv python:

    CUDA_VISIBLE_DEVICES=1 gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
        -m pytest multiview/tests -q
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from robocasa.utils.gym_utils.gymnasium_basic import create_env_robosuite  # noqa: E402

import multiview_tasks  # noqa: E402
from multiview_tasks.cameras import CAMERA_PROFILES, MOUNTED_CAMERAS  # noqa: E402
from multiview_tasks.registry import ROBOT, env_id  # noqa: E402
from multiview_tasks.tasks import TASK_ENV_NAMES  # noqa: E402
from multiview_tasks.tasks.blind_insertion import BlindInsertionBottleCabinet  # noqa: E402

BLIND = TASK_ENV_NAMES[BlindInsertionBottleCabinet.TASK_NAME]


def make_robosuite(env_name, seed):
    # Object sampling also draws from the global numpy rng (RoboCasaEnv.reset seeds it).
    np.random.seed(seed)
    env, _ = create_env_robosuite(
        env_name=env_name,
        robots=ROBOT,
        camera_names=["egoview"],
        camera_widths=64,
        camera_heights=64,
        seed=seed,
    )
    return env


def scene_state(env):
    """Object pose, door state, layout and style: what must match between paired episodes."""
    obj_body = env.sim.model.body_name2id(env.objects["obj"].root_body)
    return dict(
        obj_pos=env.sim.data.body_xpos[obj_body].copy(),
        door=env.cabinet.get_door_state(env=env)["door"],
        layout=env.layout_id,
        style=env.style_id,
    )


def test_all_variants_registered():
    for env_names in TASK_ENV_NAMES.values():
        for env_name in env_names.values():
            for profile in CAMERA_PROFILES:
                assert env_id(env_name, profile) in multiview_tasks.REGISTERED_IDS
                gym.spec(env_id(env_name, profile))


@pytest.mark.parametrize("seed", [0, 1])
def test_level0_reproduces_original_task(seed):
    original = make_robosuite("PnPBottleToCabinetClose", seed)
    variant = make_robosuite(BLIND[0], seed)
    a, b = scene_state(original), scene_state(variant)
    original.close()
    variant.close()
    np.testing.assert_allclose(a["obj_pos"], b["obj_pos"])
    assert a["door"] == pytest.approx(b["door"])
    assert (a["layout"], a["style"]) == (b["layout"], b["style"])


def test_levels_set_door_range():
    for level, env_name in BLIND.items():
        env = make_robosuite(env_name, seed=0)
        lo, hi = BlindInsertionBottleCabinet.LEVELS[level]["door_open"]
        door = env.cabinet.get_door_state(env=env)["door"]
        env.close()
        assert lo - 1e-6 <= door <= hi + 1e-6, (level, door)


def test_mounted_cameras_exist():
    env = make_robosuite(BLIND[0], seed=0)
    names = {env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)}
    env.close()
    assert {cam.name for cam in MOUNTED_CAMERAS} <= names


def test_phase_signals_match_success_check():
    env = make_robosuite(BLIND[0], seed=0)
    phases = env.phase_signals()
    assert not phases["inserted"] and not phases["door_closed"]

    # Teleport the bottle to the cabinet interior center and close the door.
    p0, px, py, pz = env.cabinet.get_int_sites(relative=False)
    center = p0 + ((px - p0) + (py - p0) + (pz - p0)) / 2
    obj_joint = env.objects["obj"].joints[0]
    env.sim.data.set_joint_qpos(obj_joint, np.concatenate([center, [1, 0, 0, 0]]))
    env.cabinet.set_door_state(min=0.0, max=0.0, env=env, rng=env.rng)
    env.sim.forward()
    phases = env.phase_signals()
    success = env._check_success()
    env.close()
    assert phases["inserted"] and phases["door_closed"] and success


def test_diagnostics_start_clear_and_detect_door_push():
    env = make_robosuite(BLIND[1], seed=0)
    diag = env.diagnostic_signals()
    assert diag and not any(diag.values()), diag
    env.cabinet.set_door_state(min=0.1, max=0.1, env=env, rng=env.rng)
    env.sim.forward()
    diag = env.diagnostic_signals()
    env.close()
    assert diag["door_pushed"]


def test_seeded_reset_pairs_scenes_across_levels():
    states = []
    for level in (0, max(BLIND)):
        env = gym.make(env_id(BLIND[level], "cam1"), enable_render=True).unwrapped
        env.reset(seed=123)
        states.append(scene_state(env.env))
        env.close()
    np.testing.assert_allclose(states[0]["obj_pos"], states[1]["obj_pos"])
    assert (states[0]["layout"], states[0]["style"]) == (states[1]["layout"], states[1]["style"])


@pytest.mark.parametrize("profile", sorted(CAMERA_PROFILES))
def test_camera_profile_observations(profile):
    env = gym.make(env_id(BLIND[0], profile), enable_render=True)
    obs, info = env.reset(seed=0)
    obs, _, _, _, info = env.step(env.action_space.sample())
    env.close()
    video_keys = {k for k in obs if k.startswith("video.")}
    expected = {cam.obs_key for cam in CAMERA_PROFILES[profile]}
    if "video.ego_view_pad_res256_freq20" in expected:
        expected.add("video.ego_view_bg_crop_pad_res256_freq20")  # added by GrootRoboCasaEnv
    assert video_keys == expected
    for key in video_keys:
        assert obs[key].shape == (256, 256, 3)
    assert {"phase/grasped", "phase/inserted", "phase/door_closed"} <= set(info)


def test_color_button_success_requires_target_only():
    from multiview_tasks.tasks.color_button import BOX_PREFIX, BUTTON_TRAVEL, COLORS

    name = TASK_ENV_NAMES["ColorButtonBox"][0]
    for press_target in (True, False):
        env = make_robosuite(name, seed=0)
        assert not env._check_success()
        target = env.target_color
        color = target if press_target else next(c for c in COLORS if c != target)
        env.sim.data.set_joint_qpos(f"{BOX_PREFIX}button_{color}_joint", BUTTON_TRAVEL)
        env.sim.forward()
        success = env._check_success()
        assert env.get_ep_meta()["lang"] == f"press the {target} button on the side of the box"
        env.close()
        assert success == press_target


# Camera table of the RoboCasa-GR1 Multiview-15views release (parent body, pos, quat
# wxyz or xyaxes, fovy) for the streams used here.
RELEASE_CAMERAS = {
    "egoview": ("robot0_head_pitch", (0.2298, 0.0, 0.0905), (0.676, 0.205, -0.205, -0.676), 77),
    "robot0_robotview": ("robot0_head_pitch", (0.13, 0.0, 0.09), "0 -1 0 0.9 0 0.6", 50),
    "left_wrist_under": (
        "robot0_left_hand_eefd", (-0.0221, 0.0411, 0.0109), (-0.3277, 0.2035, 0.1928, 0.9022), 90
    ),
    "right_wrist_under": (
        "robot0_right_hand_eef", (-0.0221, -0.0411, 0.0109), (0.9022, 0.1928, 0.2035, -0.3277), 90
    ),
    "shoulder_left": (
        "robot0_left_arm_mount", (0.1002, 0.0847, 0.1811), (0.7051, 0.0539, -0.3468, -0.6162), 77
    ),
    "shoulder_right": (
        "robot0_right_arm_mount", (0.1002, -0.0843, 0.1813), (0.6162, 0.3468, -0.0539, -0.7050), 77
    ),
}


def _rotation(orientation):
    from scipy.spatial.transform import Rotation

    if isinstance(orientation, str):  # MJCF xyaxes: camera x and y axes in the parent frame
        values = np.array(orientation.split(), dtype=float)
        x = values[:3] / np.linalg.norm(values[:3])
        y = values[3:] - np.dot(values[3:], x) * x
        y /= np.linalg.norm(y)
        return Rotation.from_matrix(np.stack([x, y, np.cross(x, y)], axis=1))
    w, x, y, z = orientation
    return Rotation.from_quat([x, y, z, w])


def test_camera_poses_match_release():
    env = make_robosuite(TASK_ENV_NAMES["ColorButtonBox"][0], seed=0)
    model = env.sim.model._model
    profile_names = {cam.name for cam in CAMERA_PROFILES["collect"]}
    assert profile_names == set(RELEASE_CAMERAS)
    for name, (parent, pos, orientation, fovy) in RELEASE_CAMERAS.items():
        cam = model.camera(name)
        assert model.body(model.cam_bodyid[cam.id]).name == parent, name
        np.testing.assert_allclose(model.cam_pos[cam.id], pos, atol=1e-4, err_msg=name)
        w, x, y, z = model.cam_quat[cam.id]
        angle = (_rotation(orientation).inv() * _rotation((w, x, y, z))).magnitude()
        assert angle < 2e-3, (name, angle)
        assert model.cam_fovy[cam.id] == pytest.approx(fovy), name
    env.close()


def test_square_framing_matches_release():
    from multiview_tasks.registry import square_frame
    import cv2

    img = np.random.default_rng(0).integers(0, 256, (800, 1280, 3), dtype=np.uint8)
    expected = cv2.resize(img[:, 240:1040], (256, 256), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(square_frame(img), expected)
