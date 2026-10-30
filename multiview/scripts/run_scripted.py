"""Run the scripted ColorButtonBox policy and report its success rate per level.

With --video_dir, each episode is saved as an H.264 mp4 with the camera views of
--profile side by side, plus results.json with the per-episode outcome.

Usage (robocasa_uv python):
    python multiview/scripts/run_scripted.py --levels 0 1 2 3 --n 20 \
        --video_dir ~/tmp/scripted_videos
"""

import argparse
import json
import os
from pathlib import Path

import av
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")
import multiview_tasks  # noqa: E402,F401  (registers gr1_mv/* envs)
from multiview_tasks.cameras import CAMERA_PROFILES  # noqa: E402
from multiview_tasks.registry import env_id  # noqa: E402
from multiview_tasks.scripted.color_button import (  # noqa: E402
    ColorButtonScriptedPolicy,
    trajectory_rng,
)
from multiview_tasks.tasks import TASK_ENV_NAMES  # noqa: E402


def tile_views(obs, cameras, caption):
    """Camera views side by side, each labeled, with a caption on the first."""
    tiles = []
    for i, cam in enumerate(cameras):
        pil = Image.fromarray(obs[cam.obs_key])
        draw = ImageDraw.Draw(pil)
        draw.text((4, 4), cam.name, fill=(255, 255, 0))
        if i == 0:
            draw.text((4, 242), caption, fill=(255, 255, 255))
        tiles.append(np.asarray(pil))
    return np.concatenate(tiles, axis=1)


def write_mp4(path, frames, fps=20):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def run_episode(env, seed, idle_steps, cameras=None, video_path=None):
    obs, info = env.reset(seed=seed)
    policy = ColorButtonScriptedPolicy(env.unwrapped)
    policy.reset(obs, rng=trajectory_rng(seed))
    record = video_path is not None
    frames = [tile_views(obs, cameras, "start")] if record else []
    errors = {}
    for _ in range(policy.total_steps + idle_steps):
        phase = policy.phase()
        obs, _, _, _, info = env.step(policy.act())
        errors.setdefault(phase, []).append(policy.tip_error())
        if record:
            frames.append(tile_views(obs, cameras, phase))
    if record:
        write_mp4(video_path, frames)
    result = {k: bool(v) for k, v in info.items() if k.startswith("phase/")}
    result["success"] = bool(env.unwrapped.env._check_success())
    result["target"] = env.unwrapped.env.target_color
    result["tip_err_mm"] = {k: round(1000 * float(np.max(v)), 1) for k, v in errors.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--idle_steps", type=int, default=10)
    parser.add_argument("--profile", default="collect", choices=sorted(CAMERA_PROFILES))
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)

    env_names = TASK_ENV_NAMES["ColorButtonBox"]
    cameras = CAMERA_PROFILES[args.profile]
    for level in args.levels:
        env = gym.make(env_id(env_names[level], args.profile), enable_render=True)
        results = []
        for i in range(args.n):
            seed = args.seed0 + i
            video = args.video_dir / f"L{level}_seed{seed}.mp4" if args.video_dir else None
            r = run_episode(env, seed, args.idle_steps, cameras, video)
            r.update(level=level, seed=seed, video=video.name if video else None)
            results.append(r)
            if args.verbose:
                print(f"L{level} seed{seed}", r, flush=True)
        env.close()
        if args.video_dir:
            with open(args.video_dir / f"results_L{level}.json", "w") as f:
                json.dump(results, f, indent=1)
        rate = lambda k: np.mean([r[k] for r in results])  # noqa: E731
        print(
            f"L{level}: success {rate('success'):.0%}  reached {rate('phase/reached'):.0%}  "
            f"pressed_wrong {rate('phase/pressed_wrong'):.0%}  (n={args.n})",
            flush=True,
        )


if __name__ == "__main__":
    main()
