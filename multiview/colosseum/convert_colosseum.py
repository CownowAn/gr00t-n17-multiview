"""Convert the ColosseumV2 dual-arm Panda demos (ManiSkill HDF5) into a LeRobot v2.0 dataset for GR00T N1.7.

Source: one HDF5 with traj_<i> groups plus the ManiSkill trajectory JSON (env_id, seed, success per episode).
Each step pairs the observation o_t with the action a_t taken from it; an episode stores T+1 observations
and T actions, so the final observation (after the last action) has no action and is dropped.

    observation.state (18): left_arm 7 | left_gripper 2 (finger widths) | right_arm 7 | right_gripper 2
    action            (16): left_arm 7 | left_gripper 1 (-1 close, 1 open) | right_arm 7 | right_gripper 1
"left" is ManiSkill's arm 1 (panda1, left_arm_tcp), "right" arm 2 (panda2). The raw qpos interleaves the
two arms joint by joint (arm1 j1, arm2 j1, arm1 j2, ...) followed by arm1's and arm2's two fingers.

Cameras (native resolution, H.264): external1 / external2 (224x224), left_hand / right_hand (128x128,
the wrist cameras of panda1 / panda2). Language: the first phrasing of the task in ColosseumV2's
language_randomizations.yaml (with --paraphrases, all phrasings; the N1.7 loader samples one per frame).

Usage (colosseum_v2 conda env):
    python multiview/colosseum/convert_colosseum.py --out /nfs/data/sohyun/data/colosseum_v2_lerobot/cv2_dual10 \
        [--skip DualArmDrawerOpen-v1 DualArmDrawerPlace-v1] [--limit 2]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

SRC = Path("/nfs/data/sohyun/data/colosseum-v2/trajectory__cv2-full__pd_joint_pos__100")
LANG_YAML = Path.home() / "project/ColosseumV2/mani_skill/envs/tasks/tabletop/colosseum_v2/language_randomizations.yaml"
FPS = 20  # ColosseumV2 control frequency (sim 100 Hz)
CHUNK = 1000
CAMERAS = {  # modality key -> HDF5 sensor name
    "external1": "external1_camera",
    "external2": "external2_camera",
    "left_hand": "panda1_hand_camera",
    "right_hand": "panda2_hand_camera",
}
STATE_LAYOUT = [("left_arm", 7), ("left_gripper", 2), ("right_arm", 7), ("right_gripper", 2)]
ACTION_LAYOUT = [("left_arm", 7), ("left_gripper", 1), ("right_arm", 7), ("right_gripper", 1)]
# Skipped by default: their PartNet-mobility assets cannot be downloaded here, so they cannot be evaluated.
DEFAULT_SKIP = ["DualArmDrawerOpen-v1", "DualArmDrawerPlace-v1"]


def state_from_qpos(q: np.ndarray) -> np.ndarray:
    """(T, 18) interleaved ManiSkill qpos -> (T, 18) in STATE_LAYOUT order."""
    return np.concatenate([q[:, 0:14:2], q[:, 14:16], q[:, 1:14:2], q[:, 16:18]], axis=1)


def write_mp4(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def layout_json(layout, key):
    out, start = {}, 0
    for name, dim in layout:
        out[name] = {"original_key": key, "start": start, "end": start + dim}
        start += dim
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip", nargs="*", default=DEFAULT_SKIP)
    ap.add_argument("--limit", type=int, default=0, help="episodes per task (0 = all); for quick tests")
    ap.add_argument("--paraphrases", action="store_true", help="store every phrasing of the instruction")
    args = ap.parse_args()

    meta = json.load(open(f"{SRC}.json"))
    lang = yaml.safe_load(open(LANG_YAML))
    episodes = [e for e in meta["episodes"] if e["success"] and e["env_id"] not in args.skip]
    if args.limit:
        per = {}
        episodes = [e for e in episodes if per.setdefault(e["env_id"], []).append(e) or len(per[e["env_id"]]) <= args.limit]

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "meta").mkdir(parents=True)
    task_index, ep_rows, total, sizes = {}, [], 0, {}
    with h5py.File(f"{SRC}.h5", "r") as f:
        for new_idx, e in enumerate(episodes):
            g = f[f"traj_{e['episode_id']}"]
            act = g["actions"][:].astype(np.float64)
            T = len(act)
            state = state_from_qpos(g["obs/agent/qpos"][:T].astype(np.float64))
            phrases = lang[e["env_id"]] if args.paraphrases else lang[e["env_id"]][:1]
            for p in phrases:
                task_index.setdefault(p, len(task_index))
            chunk = new_idx // CHUNK
            for key, sensor in CAMERAS.items():
                frames = g[f"obs/sensor_data/{sensor}/rgb"][:T]
                sizes[key] = list(frames.shape[1:])
                write_mp4(args.out / f"videos/chunk-{chunk:03d}/observation.images.{key}/episode_{new_idx:06d}.mp4", frames)
            df = pd.DataFrame({
                "observation.state": list(state),
                "action": list(act),
                "timestamp": np.arange(T) / FPS,
                "next.reward": np.zeros(T),
                "next.done": np.arange(T) == T - 1,
                "task_index": np.full(T, task_index[phrases[0]]),
                "episode_index": np.full(T, new_idx),
                "index": np.arange(total, total + T),
            })
            (args.out / f"data/chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
            df.to_parquet(args.out / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet")
            ep_rows.append({"episode_index": new_idx, "tasks": phrases, "length": T, "env_id": e["env_id"],
                            "seed": e["episode_seed"], "source_episode_id": e["episode_id"]})
            total += T
            if new_idx % 50 == 0:
                print(f"{new_idx + 1}/{len(episodes)} {e['env_id']} T={T}", flush=True)

    n = len(episodes)
    video = lambda shape: {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"],  # noqa: E731
                           "video_info": {"video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                          "video.is_depth_map": False, "has_audio": False}}
    vec = lambda d: {"dtype": "float64", "shape": [d], "names": [f"motor_{i}" for i in range(d)]}  # noqa: E731
    one = lambda t: {"dtype": t, "shape": [1]}  # noqa: E731
    features = {f"observation.images.{k}": video(sizes[k]) for k in CAMERAS}
    features.update({"observation.state": vec(18), "action": vec(16), "timestamp": one("float64"),
                     "next.reward": one("float64"), "next.done": one("bool"), "task_index": one("int64"),
                     "episode_index": one("int64"), "index": one("int64")})
    info = {"codebase_version": "v2.0", "robot_type": "dual_panda_wristcam", "total_episodes": n,
            "total_frames": total, "total_tasks": len(task_index), "total_videos": n * len(CAMERAS),
            "total_chunks": (n + CHUNK - 1) // CHUNK, "chunks_size": CHUNK, "fps": float(FPS),
            "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features}
    modality = {"state": layout_json(STATE_LAYOUT, "observation.state"), "action": layout_json(ACTION_LAYOUT, "action"),
                "video": {k: {"original_key": f"observation.images.{k}"} for k in CAMERAS}}
    m = args.out / "meta"
    json.dump(info, open(m / "info.json", "w"), indent=4)
    json.dump(modality, open(m / "modality.json", "w"), indent=4)
    json.dump({"robot_type": "dual_panda_wristcam", "embodiment_tag": "new_embodiment"}, open(m / "embodiment.json", "w"), indent=4)
    with open(m / "tasks.jsonl", "w") as fh:
        for p, i in task_index.items():
            fh.write(json.dumps({"task_index": i, "task": p}) + "\n")
    with open(m / "episodes.jsonl", "w") as fh:
        for r in ep_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}: {n} episodes, {total} frames, tasks {sorted({r['env_id'] for r in ep_rows})}")


if __name__ == "__main__":
    main()
