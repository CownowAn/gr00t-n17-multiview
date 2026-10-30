"""Replay demonstrations from an exported HDF5 file and re-render their camera streams.

--mode states  restores the recorded MuJoCo state at every step (use this to render
               new or changed cameras for existing demos). Renders match the recorded
               streams to ~0.01/255 mean, except fast-moving wrist views (~1/255 mean):
               frames rendered right after env.step use kinematics one physics substep
               (2 ms) older than the recorded state.
--mode actions restores only the first state, then executes the recorded actions
               open-loop and reports success and the drift from the recorded states.

Usage (robocasa_uv python, repo root):
    python multiview/scripts/replay_hdf5.py \
        --hdf5 /nfs/data/sohyun/data/multiview_gr1/ColorButtonBox/hdf5/ColorButtonBox_L0.hdf5 \
        --demos 1 2 --out_dir ~/tmp/replay
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import multiview_tasks  # noqa: E402,F401  (registers the task classes)
from multiview_tasks.cameras import CAMERA_PROFILES  # noqa: E402
from multiview_tasks.lerobot_io import video_streams, write_mp4  # noqa: E402
from multiview_tasks.replay_io import make_env, observe_videos, reset_to, set_state  # noqa: E402


def replay_demo(env, demo, mode, profile):
    states, actions = demo["states"][()], demo["actions"][()]
    reset_to(env, demo.attrs["model_file"], json.loads(demo.attrs["ep_meta"]), states[0])
    frames, drift, success = [], 0.0, False
    for t in range(len(actions)):
        if mode == "states":
            set_state(env, states[t])
        else:
            drift = max(drift, float(np.abs(env.sim.get_state().flatten() - states[t]).max()))
        frames.append(observe_videos(env, profile))
        if mode == "actions":
            env.step(actions[t])
        # Called every step: the task latches button presses inside its success check.
        success = bool(env._check_success())
    return frames, success, drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--demos", type=int, nargs="*", default=None, help="demo_<n> numbers (default: all)")
    parser.add_argument("--mode", choices=["states", "actions"], default="states")
    parser.add_argument("--profile", default="collect", choices=sorted(CAMERA_PROFILES))
    parser.add_argument("--out_dir", type=Path, default=None, help="write one mp4 per stream")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        cameras = [cam.name for cam in CAMERA_PROFILES[args.profile]]
        env = make_env(env_meta, camera_names=cameras)
        demo_ids = args.demos or sorted(int(k.split("_")[1]) for k in f["data"])
        for i in demo_ids:
            demo = f[f"data/demo_{i}"]
            frames, success, drift = replay_demo(env, demo, args.mode, args.profile)
            print(
                f"demo_{i} (seed {demo.attrs['seed']}, episode {demo.attrs['lerobot_episode_index']}): "
                f"{len(frames)} steps, success {success}"
                + (f", max state drift {drift:.2e}" if args.mode == "actions" else ""),
                flush=True,
            )
            if args.out_dir:
                out = args.out_dir / f"demo_{i}"
                out.mkdir(parents=True, exist_ok=True)
                for obs_key, lerobot_key, _, pix_fmt in video_streams(args.profile):
                    write_mp4(out / f"{lerobot_key}.mp4", [fr[obs_key] for fr in frames], pix_fmt)
        env.close()


if __name__ == "__main__":
    main()
