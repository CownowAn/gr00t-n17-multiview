"""LeRobot v2.0 layout for multi-view GR1 demonstrations, matching gr1_unified.* datasets.

State and action are the 44-dim GR1 vectors of the released RoboCasa GR1 datasets:
parts in alphabetical order, with legs and neck (not actuated here) left at zero.
"""

import json
from pathlib import Path

import av
import numpy as np

from multiview_tasks.cameras import CAMERA_PROFILES

FPS = 20
# (part, dim) in the order they are concatenated into observation.state / action.
GR1_LAYOUT = (
    ("left_arm", 7),
    ("left_hand", 6),
    ("left_leg", 6),
    ("neck", 3),
    ("right_arm", 7),
    ("right_hand", 6),
    ("right_leg", 6),
    ("waist", 3),
)
GR1_DIM = sum(d for _, d in GR1_LAYOUT)
ZERO_PARTS = ("left_leg", "neck", "right_leg")

# egoview is saved in two framings. "bg crop" is what the base GR1 checkpoint reads and
# what the Multiview-15views dataset stores as ego_view; "pad" is robocasa's letterbox.
EGO_STREAMS = (
    ("video.ego_view_bg_crop_pad_res256_freq20", "ego_view"),
    ("video.ego_view_pad_res256_freq20", "ego_view_pad"),
)
# Pixel formats of the Multiview-15views dataset: ego_view yuv420p, all others yuv444p.
PIX_FMT = {"ego_view": "yuv420p"}
DEFAULT_PIX_FMT = "yuv444p"


def video_streams(profile: str = "collect"):
    """[(env obs key, LeRobot key, modality key, pix_fmt)] of the streams saved for a profile.

    Modality keys are the env obs keys without "video.", so a policy trained on the
    dataset reads the same keys the eval env returns.
    """
    named = []
    for cam in CAMERA_PROFILES[profile]:
        if cam.name == "egoview":
            named.extend(EGO_STREAMS)
        else:
            named.append((cam.obs_key, cam.obs_key[len("video.") :]))
    return [
        (obs_key, f"observation.images.{name}", obs_key[len("video.") :], PIX_FMT.get(name, DEFAULT_PIX_FMT))
        for obs_key, name in named
    ]


def pack_gr1(parts: dict, prefix: str) -> np.ndarray:
    """Concatenate ``{prefix}{part}`` entries into the 44-dim GR1 vector."""
    out = []
    for part, dim in GR1_LAYOUT:
        if part in ZERO_PARTS:
            out.append(np.zeros(dim))
        else:
            value = np.asarray(parts[f"{prefix}{part}"], dtype=np.float64)
            assert value.shape == (dim,), (part, value.shape)
            out.append(value)
    return np.concatenate(out)


def write_mp4(path: Path, frames, pix_fmt: str, fps: int = FPS, crf: int = 18):
    """H.264, as in the released GR1 datasets."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = pix_fmt
        stream.options = {"crf": str(crf)}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def modality_json(streams) -> dict:
    layout, start = {}, 0
    for part, dim in GR1_LAYOUT:
        layout[part] = {"original_key": None, "start": start, "end": start + dim}
        start += dim
    state = {k: dict(v, original_key="observation.state") for k, v in layout.items()}
    action = {k: dict(v, original_key="action") for k, v in layout.items()}
    return {
        "state": state,
        "action": action,
        "video": {mod: {"original_key": lr} for _, lr, mod, _ in streams},
        "annotation": {"human.coarse_action": {"original_key": "annotation.human.coarse_action"}},
    }


def info_json(streams, total_episodes: int, total_frames: int, total_tasks: int, chunks_size: int):
    def video_feature(pix_fmt):
        return {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(FPS),
                "video.codec": "h264",
                "video.pix_fmt": pix_fmt,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    vector = {"dtype": "float64", "shape": [GR1_DIM], "names": [f"motor_{i}" for i in range(GR1_DIM)]}
    scalar = lambda dtype: {"dtype": dtype, "shape": [1]}  # noqa: E731
    features = {lr: video_feature(pix_fmt) for _, lr, _, pix_fmt in streams}
    features.update(
        {
            "observation.state": vector,
            "action": vector,
            "timestamp": scalar("float64"),
            "next.reward": scalar("float64"),
            "next.done": scalar("bool"),
            "task_index": scalar("int64"),
            "annotation.human.coarse_action": scalar("int64"),
            "episode_index": scalar("int64"),
            "index": scalar("int64"),
        }
    )
    return {
        "codebase_version": "v2.0",
        "robot_type": "GR1ArmsAndWaistFourierHands",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(streams),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": float(FPS),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
