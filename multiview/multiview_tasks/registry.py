"""Gym registration for multi-view task variants.

IDs look like ``gr1_mv/<Task>_L<level>_<profile>``, e.g.
``gr1_mv/BlindInsertionBottleCabinet_L2_cam3``. Encoding level and camera profile
in the ID lets gr00t/eval/rollout_policy.py run any variant through --env_name.
"""

import json
import os

import cv2
import numpy as np
from gymnasium.envs.registration import register
from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv

from multiview_tasks.cameras import CAMERA_HEIGHT, CAMERA_PROFILES, CAMERA_WIDTH

ENV_PREFIX = "gr1_mv"
ROBOT = "GR1ArmsAndWaistFourierHands"


def square_frame(img: np.ndarray) -> np.ndarray:
    """Centre square crop (full height) resized to 256x256: no padding, no distortion."""
    h, w, _ = img.shape
    x0 = (w - h) // 2
    return cv2.resize(img[:, x0 : x0 + h], (256, 256), interpolation=cv2.INTER_AREA)


def _with_camera_profile(key_converter: type, profile: str) -> type:
    """Return a key converter whose camera config is the given profile."""
    cameras = CAMERA_PROFILES[profile]

    def get_camera_config(cls):
        return (
            [cam.obs_key for cam in cameras],
            [cam.name for cam in cameras],
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )

    return type(
        f"{key_converter.__name__}_{profile}",
        (key_converter,),
        {"get_camera_config": classmethod(get_camera_config)},
    )


class MultiViewGrootEnv(GrootRoboCasaEnv):
    """GrootRoboCasaEnv that renders and returns the cameras of a camera profile.

    RoboCasaEnv reads the camera list from ``self.key_converter`` while it builds the
    robosuite env, so the profile is applied when the converter is assigned.
    """

    def __init__(self, camera_profile: str = "cam1", eval_seed=None, **kwargs):
        """eval_seed: (base, env_idx, n_envs). Unseeded resets then use base + env_idx +
        k * n_envs for the k-th episode, so parallel evals of different checkpoints or
        camera profiles see the same scenes."""
        if camera_profile not in CAMERA_PROFILES:
            raise ValueError(
                f"Unknown camera profile {camera_profile!r}; "
                f"expected one of {list(CAMERA_PROFILES)}"
            )
        self.camera_profile = camera_profile
        self._eval_seed = eval_seed
        self._n_resets = 0
        self._episode_seed = -1
        # Episode log: with GR1_MV_EPISODE_LOG_DIR set, every finished episode of a task that
        # defines episode_record_header() / episode_record_step() is appended as one JSON line
        # (seed, per-step values, final phase/diag flags) to <dir>/episodes_<pid>.jsonl.
        self._log_dir = os.environ.get("GR1_MV_EPISODE_LOG_DIR")
        self._record = None
        super().__init__(**kwargs)

    @property
    def key_converter(self):
        return self._key_converter

    @key_converter.setter
    def key_converter(self, converter):
        self._key_converter = _with_camera_profile(converter, self.camera_profile)

    def get_groot_observation(self, raw_obs):
        obs = super().get_groot_observation(raw_obs)
        # The parent letterboxes every camera; re-frame the "square" ones.
        for cam in CAMERA_PROFILES[self.camera_profile]:
            if cam.framing == "square":
                obs[cam.obs_key] = square_frame(raw_obs[f"{cam.name}_image"])
        return obs

    def _episode_signals(self) -> dict:
        # Each task variant defines its own flags (see e.g. BlindInsertionBottleCabinet):
        # phase_signals() for task progress, diagnostic_signals() for failure analysis.
        signals = {}
        for prefix, method in (("phase", "phase_signals"), ("diag", "diagnostic_signals")):
            if hasattr(self.env, method):
                for name, value in getattr(self.env, method)().items():
                    signals[f"{prefix}/{name}"] = bool(value)
        return signals

    def _add_phase_info(self, info: dict) -> None:
        """Report each flag as reached-at-any-point-this-episode (``phase/*``, ``diag/*``)."""
        for key, value in self._episode_signals().items():
            self._phases_reached[key] = self._phases_reached.get(key, False) or value
        info.update(self._phases_reached)
        # metric/*: episode min / max of the task's per-step numbers (metric_signals()).
        if hasattr(self.env, "metric_signals"):
            for name, value in self.env.metric_signals().items():
                key = f"metric/{name}"
                prev = self._phases_reached.get(key)
                agg = min if name.startswith("min_") else max
                self._phases_reached[key] = value if prev is None else agg(prev, value)
            info.update({k: v for k, v in self._phases_reached.items() if k.startswith("metric/")})
        info["episode_seed"] = self._episode_seed

    def _flush_record(self):
        if self._record is not None and self._record["steps"]:
            self._record["final"] = {k: v for k, v in self._phases_reached.items() if not k.startswith("metric/")}
            self._record["success"] = bool(self.env._check_success())
            os.makedirs(self._log_dir, exist_ok=True)
            with open(os.path.join(self._log_dir, f"episodes_{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps(self._record) + "\n")
        self._record = None

    def close(self):
        if self._log_dir:
            self._flush_record()
        super().close()

    def reset(self, seed=None, options=None):
        if self._log_dir:
            self._flush_record()
        if seed is None and self._eval_seed is not None:
            base, idx, n = self._eval_seed
            seed = base + idx + self._n_resets * n
        self._n_resets += 1
        self._episode_seed = -1 if seed is None else int(seed)
        if seed is not None:
            # robocasa samples layout, style and placements from the robosuite env rng, which
            # RoboCasaEnv never seeds. Reseeding it pairs episodes across levels and profiles.
            self.env.rng = np.random.default_rng(seed)
        obs, info = super().reset(seed=seed, options=options)
        # Tasks with LOCK_WAIST keep the torso (and so the head camera) at its start pose:
        # any waist action is replaced, so the policy cannot lean or twist to look around.
        self._waist_hold = (
            np.asarray(obs["state.waist"], dtype=np.float64).copy()
            if getattr(self.env, "LOCK_WAIST", False)
            else None
        )
        self._phases_reached = {}
        self._add_phase_info(info)
        if self._log_dir and hasattr(self.env, "episode_record_header"):
            self._record = dict(seed=self._episode_seed, env=type(self.env).__name__,
                                **self.env.episode_record_header(), steps=[])
        return obs, info

    def step(self, action):
        if self._waist_hold is not None and "action.waist" in action:
            action = {**action, "action.waist": self._waist_hold}
        obs, reward, terminated, truncated, info = super().step(action)
        self._add_phase_info(info)
        if self._record is not None:
            self._record["steps"].append(self.env.episode_record_step())
        return obs, reward, terminated, truncated, info


def env_id(env_name: str, profile: str) -> str:
    return f"{ENV_PREFIX}/{env_name}_{profile}"


def register_variants(env_names: dict) -> list:
    """Register one gym ID per (level, camera profile). Returns the registered IDs."""
    ids = []
    for env_name in env_names.values():
        for profile in CAMERA_PROFILES:
            gym_id = env_id(env_name, profile)
            register(
                id=gym_id,
                entry_point="multiview_tasks.registry:MultiViewGrootEnv",
                kwargs=dict(env_name=env_name, robots_name=ROBOT, camera_profile=profile),
            )
            ids.append(gym_id)
    return ids
