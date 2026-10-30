"""Save what every camera sees at reset, for each level of a task (design check).

Images are the processed 256x256 observations the policy receives. One row per
level, one column per camera in the collect profile.

Usage (robocasa_uv python):
    python multiview/scripts/render_views.py --task BlindInsertionBottleCabinet \
        --seeds 0 1 2 --out_dir /tmp/mv_views
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")
import multiview_tasks  # noqa: E402,F401  (registers gr1_mv/* envs)
from multiview_tasks.cameras import CAMERA_PROFILES  # noqa: E402
from multiview_tasks.registry import env_id  # noqa: E402
from multiview_tasks.tasks import TASK_ENV_NAMES  # noqa: E402


def label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(pil)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_ENV_NAMES))
    parser.add_argument("--levels", type=int, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    env_names = TASK_ENV_NAMES[args.task]
    levels = args.levels if args.levels is not None else sorted(env_names)
    cameras = CAMERA_PROFILES["collect"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        rows = []
        for level in levels:
            env = gym.make(env_id(env_names[level], "collect"), enable_render=True)
            obs, _ = env.reset(seed=seed)
            lang = obs["annotation.human.coarse_action"]
            tiles = [label(obs[cam.obs_key], f"L{level} {cam.name}") for cam in cameras]
            rows.append(np.concatenate(tiles, axis=1))
            env.close()
        path = args.out_dir / f"{args.task}_seed{seed}.png"
        Image.fromarray(np.concatenate(rows, axis=0)).save(path)
        print(f"saved {path}  ({lang})")


if __name__ == "__main__":
    main()
