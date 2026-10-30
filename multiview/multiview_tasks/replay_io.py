"""Replayable demonstrations in the robomimic / robocasa HDF5 layout.

Follows the released GR1 teleop HDF5 files (PhysicalAI-Robotics-GR00T-Teleop-Sim):
    data                    attrs: env_args (json: type, env_name, env_version,
                                   robosuite_version, mujoco_version, env_kwargs),
                                   env_info (json env_kwargs), env, date, time,
                                   robocasa_version, robosuite_version, mujoco_version, total
    data/demo_<n>           n = 1, 2, ... ; attrs: model_file (scene xml), ep_meta (json),
                                   num_samples, plus seed and lerobot_episode_index (= n - 1)
        states              (T, S) flattened MuJoCo state before each step
        actions             (T, A) flat robosuite action passed to env.step
        action_dict/<part>  the action split by GR1 part (joint positions)
        gr1_actions         (T, 44) the action in the GR1 LeRobot layout
        gr1_states          (T, 44) proprioceptive state in the GR1 LeRobot layout
The released files were recorded with the WHOLE_BODY_MINK_IK controller, so their
actions and action_dict are end-effector targets; these demos use the joint-position
controller of the GR00T eval env, recorded in env_args.

Restoring model_file + ep_meta + states[t] reproduces the scene at step t exactly, so
any camera can be re-rendered later (robocasa's playback_dataset.py reset logic).
"""

import json
import time

import mujoco
import numpy as np
import robocasa
import robosuite

from multiview_tasks.lerobot_io import GR1_LAYOUT, ZERO_PARTS

ENV_TYPE_ROBOSUITE = 1  # robomimic EnvType.ROBOSUITE_TYPE


def flat_env_action(gym_env, action: dict) -> np.ndarray:
    """The flat robosuite action that RoboCasaEnv.step builds from a GR00T action dict."""
    raw = gym_env.key_converter.unmap_action(dict(action))
    robot = gym_env.env.robots[0]
    cc = robot.composite_controller
    prefix = robot.robot_model.naming_prefix
    flat = np.zeros(cc.action_limits[0].shape)
    for part, (start, end) in cc._action_split_indexes.items():
        flat[start:end] = raw[f"{prefix}{part}"]
    return flat


def env_args(gym_env) -> dict:
    """robomimic env metadata: robosuite.make(**env_kwargs) rebuilds the env."""
    kwargs = dict(gym_env.env_kwargs)
    return {
        "type": ENV_TYPE_ROBOSUITE,
        "env_name": kwargs["env_name"],
        "env_version": robocasa.__version__,
        "robosuite_version": robosuite.__version__,
        "mujoco_version": mujoco.__version__,
        "env_kwargs": kwargs,
    }


def make_env(env_meta: dict, camera_names=None):
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.pop("env_name", None)
    if camera_names is not None:
        kwargs["camera_names"] = list(camera_names)
    return robosuite.make(env_meta["env_name"], **kwargs)


def reset_to(env, model_xml: str, ep_meta: dict, state: np.ndarray):
    """Load a recorded scene and set its MuJoCo state (robocasa playback_dataset.reset_to)."""
    env.set_ep_meta(ep_meta)
    env.reset()  # hard reset; reset_from_xml_string alone does not reload the model
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()
    set_state(env, state)


def set_state(env, state: np.ndarray):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def observe_videos(env, profile: str) -> dict:
    """256x256 video observations of a camera profile, as MultiViewGrootEnv returns them."""
    from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv

    from multiview_tasks.cameras import CAMERA_PROFILES
    from multiview_tasks.registry import square_frame

    raw = env._get_observations(force_update=True)
    videos = {}
    for cam in CAMERA_PROFILES[profile]:
        img = np.copy(raw[f"{cam.name}_image"][::-1])  # RoboCasaEnv flips renders upright
        if cam.framing == "pad":
            videos[cam.obs_key] = GrootRoboCasaEnv.process_img(img)
            videos["video.ego_view_bg_crop_pad_res256_freq20"] = GrootRoboCasaEnv.process_img_cotrain(img)
        else:
            videos[cam.obs_key] = square_frame(img)
    return videos


def write_hdf5(path, env_meta: dict, demos):
    """``demos``: iterable of dicts with model_file, ep_meta, seed, lerobot_episode_index,
    states, actions, gr1_actions, gr1_states."""
    import h5py

    now = time.localtime()
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["date"] = time.strftime("%m-%d-%Y", now)
        data.attrs["time"] = time.strftime("%H:%M:%S", now)
        data.attrs["env"] = env_meta["env_name"]
        data.attrs["env_args"] = json.dumps(env_meta, indent=4)
        data.attrs["env_info"] = json.dumps(env_meta["env_kwargs"])
        data.attrs["robocasa_version"] = env_meta["env_version"]
        data.attrs["robosuite_version"] = env_meta["robosuite_version"]
        data.attrs["mujoco_version"] = env_meta["mujoco_version"]
        total = 0
        for demo in demos:
            group = data.create_group(f"demo_{demo['lerobot_episode_index'] + 1}")
            group.attrs["model_file"] = demo["model_file"]
            group.attrs["ep_meta"] = json.dumps(demo["ep_meta"])
            group.attrs["num_samples"] = len(demo["actions"])
            group.attrs["seed"] = demo["seed"]
            group.attrs["lerobot_episode_index"] = demo["lerobot_episode_index"]
            for key in ("states", "actions", "gr1_actions", "gr1_states"):
                group.create_dataset(key, data=demo[key], compression="gzip")
            start = 0
            for part, dim in GR1_LAYOUT:
                if part not in ZERO_PARTS:
                    group.create_dataset(f"action_dict/{part}", data=demo["gr1_actions"][:, start : start + dim])
                start += dim
            total += len(demo["actions"])
        data.attrs["total"] = total
