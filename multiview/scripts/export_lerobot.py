"""Convert staged scripted episodes of one level into a LeRobot v2.0 dataset and HDF5.

Takes the --n successful episodes with the lowest seeds from <out_root>/staging/L<level>
(with --paired_levels, only seeds that succeeded at all those levels) and writes
    <out_root>/lerobot/gr1_mv.<task>_L<level>   LeRobot v2.0 (training)
    <out_root>/hdf5/<task>_L<level>.hdf5        robomimic layout (replay; demo_<n>
                                                        is LeRobot episode_index n - 1)
Afterwards compute the normalization stats GR00T requires (main GR00T venv):
    .venv/bin/python gr00t/data/stats.py <dataset> --embodiment-tag gr1

Usage (robocasa_uv python, repo root):
    python multiview/scripts/export_lerobot.py --level 0 --n 200 \
        --out_root /nfs/data/sohyun/data/multiview_gr1/ColorButtonBox
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from multiview_tasks.lerobot_io import (
    FPS,
    info_json,
    modality_json,
    video_streams,
    write_jsonl,
)
from multiview_tasks.replay_io import write_hdf5

CHUNK_SIZE = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", type=int, nargs="+", required=True,
        help="one level, or several to merge into one dataset (LeRobot only, no HDF5)",
    )
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--task", default="ColorButtonBox", help="task name used in the output names")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--profile", default="collect")
    parser.add_argument(
        "--paired_levels", type=int, nargs="*", default=None,
        help="use only seeds that succeeded at all of these levels (e.g. 0 1 2 3)",
    )
    args = parser.parse_args()

    common = None
    if args.paired_levels:
        # Keep only seeds that succeeded at every listed level, so the datasets of
        # different levels hold the same scenes and trajectories.
        common = set.intersection(
            *(
                {p.name for p in (args.out_root / "staging" / f"L{level}").glob("seed_*") if p.is_dir()}
                for level in args.paired_levels
            )
        )
    episodes = []  # --n episodes per level, level by level
    for level in args.level:
        staging = args.out_root / "staging" / f"L{level}"
        level_eps = sorted(p for p in staging.glob("seed_*") if p.is_dir())
        if common is not None:
            level_eps = [p for p in level_eps if p.name in common]
        if len(level_eps) < args.n:
            raise SystemExit(f"{staging} has {len(level_eps)} usable successful episodes, need {args.n}")
        episodes.extend(level_eps[: args.n])

    name = "L" + "".join(str(level) for level in args.level)
    out = args.out_root / "lerobot" / f"gr1_mv.{args.task}_{name}"
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    streams = video_streams(args.profile)

    tasks, episode_rows, demos, index = {}, [], [], 0
    for ep_idx, ep_dir in enumerate(episodes):
        meta = json.loads((ep_dir / "meta.json").read_text())
        lowdim = np.load(ep_dir / "lowdim.npz")
        length = len(lowdim["action"])
        task_idx = tasks.setdefault(meta["language"], len(tasks))
        chunk = ep_idx // CHUNK_SIZE

        done = np.zeros(length, dtype=bool)
        done[-1] = True
        df = pd.DataFrame(
            {
                "observation.state": list(lowdim["state"]),
                "action": list(lowdim["action"]),
                "timestamp": np.arange(length, dtype=np.float64) / FPS,
                "next.reward": done.astype(np.float64),
                "next.done": done,
                "task_index": np.full(length, task_idx, dtype=np.int64),
                "annotation.human.coarse_action": np.full(length, task_idx, dtype=np.int64),
                "episode_index": np.full(length, ep_idx, dtype=np.int64),
                "index": np.arange(index, index + length, dtype=np.int64),
            }
        )
        data_dir = out / "data" / f"chunk-{chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet")
        for _, lerobot_key, _, _ in streams:
            video_dir = out / "videos" / f"chunk-{chunk:03d}" / lerobot_key
            video_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ep_dir / f"{lerobot_key}.mp4", video_dir / f"episode_{ep_idx:06d}.mp4")

        episode_rows.append(
            {
                "episode_index": ep_idx,
                "tasks": [meta["language"]],
                "length": length,
                "seed": meta["seed"],
                "level": meta["level"],
                **{k: meta[k] for k in ("target", "button_colors", "face_side", "block_colors") if k in meta},
            }
        )
        demos.append(
            {
                "model_file": gzip.open(ep_dir / "model.xml.gz", "rt").read(),
                "ep_meta": meta["ep_meta"],
                "seed": meta["seed"],
                "lerobot_episode_index": ep_idx,
                "states": lowdim["sim_states"],
                "actions": lowdim["env_actions"],
                "gr1_actions": lowdim["action"],
                "gr1_states": lowdim["state"],
            }
        )
        index += length

    meta_dir = out / "meta"
    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in tasks.items()])
    (meta_dir / "modality.json").write_text(json.dumps(modality_json(streams), indent=4))
    (meta_dir / "info.json").write_text(
        json.dumps(info_json(streams, len(episodes), index, len(tasks), CHUNK_SIZE), indent=4)
    )
    (meta_dir / "embodiment.json").write_text(
        json.dumps({"robot_type": "GR1ArmsAndWaistFourierHands", "embodiment_tag": "gr1"}, indent=4)
    )
    outputs = [out]
    if len(args.level) == 1:  # replay files are per level (env_args name the level)
        hdf5_path = args.out_root / "hdf5" / f"{args.task}_{name}.hdf5"
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        write_hdf5(hdf5_path, json.loads((staging / "env_args.json").read_text()), demos)
        outputs.append(hdf5_path)

    print(
        f"wrote {', '.join(map(str, outputs))}: {len(episodes)} episodes, {index} frames, "
        f"{len(tasks)} tasks"
    )


if __name__ == "__main__":
    main()
