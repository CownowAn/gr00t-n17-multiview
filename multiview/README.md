# multiview_tasks

Observability variants of RoboCasa GR1 tabletop tasks, for testing whether extra
camera views (wrist-under, shoulder) help a GR00T policy.

- A variant is an existing robocasa task plus one knob. **Level 0 reproduces the
  original task** (same layout, language, success check).
- Every scene carries all mounted cameras; the camera profile only decides which
  are rendered and returned: `cam1` (egoview, the base checkpoint's input),
  `cam3` (+ wrist-under views), `cam5` (+ shoulder views); `collect` (all streams,
  incl. robotview) is for datasets. Wrist-under and shoulder poses are those of the
  RoboCasa-GR1 Multiview-15views release; their 256x256 observations use its
  "square" framing (centre 800x800 crop, INTER_AREA resize).
- Gym IDs: `gr1_mv/<Task>_L<level>_<profile>`, usable directly as
  `--env_name` in `gr00t/eval/rollout_policy.py`.

## Layout

| Path | Purpose |
| --- | --- |
| `multiview_tasks/cameras.py` | Camera mounts, framing, obs keys and profiles (single source of truth) |
| `multiview_tasks/camera_mixin.py` | Adds the mounted cameras to a Tabletop scene |
| `multiview_tasks/variant.py` | Level parameters; one robosuite class per level |
| `multiview_tasks/tasks/` | Task variants |
| `multiview_tasks/registry.py` | `MultiViewGrootEnv` (camera profile, seeding, `phase/*` info) and gym registration |
| `multiview_tasks/scripted/` | mink-IK scripted policies that generate demonstrations |
| `scripts/run_scripted.py` | Runs a scripted policy, reports success, saves tiled-view videos |
| `multiview_tasks/lerobot_io.py` | GR1 44-dim state/action layout, video streams, LeRobot meta |
| `scripts/collect_scripted.py` | Collects successful demos for one level on one GPU (resumable, restarts stalled workers) |
| `multiview_tasks/replay_io.py` | Replayable HDF5 (robomimic layout): write, reset to a recorded state, re-render |
| `scripts/export_lerobot.py` | Staged demos -> LeRobot v2.0 dataset (`gr1_mv.ColorButtonBox_L<n>`) + HDF5 |
| `scripts/replay_hdf5.py` | Replays HDF5 demos (exact states or open-loop actions), re-renders any camera profile |
| `scripts/render_views.py` | Saves every camera's view per level (design check) |
| `tests/` | Level 0 == original, paired seeds, camera profiles |

## Setup and usage

All commands run from the repo root with the sim env python:

```bash
PY=gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python
uv pip install --python $PY --no-deps -e multiview --config-settings editable_mode=compat

CUDA_VISIBLE_DEVICES=1 $PY -m pytest multiview/tests -q
CUDA_VISIBLE_DEVICES=1 $PY multiview/scripts/render_views.py \
    --task BlindInsertionBottleCabinet --seeds 0 1 --out_dir /tmp/mv_views
```

Evaluate a variant with the base checkpoint (server as usual, egoview only = `cam1`):

```bash
CUDA_VISIBLE_DEVICES=1 $PY gr00t/eval/rollout_policy.py \
    --env_name gr1_mv/BlindInsertionBottleCabinet_L1_cam1 \
    --policy_client_host 127.0.0.1 --policy_client_port 5555 \
    --n_episodes 50 --n_envs 5 --n_action_steps 8 --max_episode_steps 720
```

Collect scripted demonstrations (one GPU per level) and convert them:

```bash
OUT=/nfs/data/sohyun/data/multiview_gr1/ColorButtonBox
$PY multiview/scripts/collect_scripted.py --level 0 --gpu 0 --n_success 200 --out_root $OUT
$PY multiview/scripts/export_lerobot.py --level 0 --n 200 --paired_levels 0 1 2 3 --out_root $OUT
.venv/bin/python gr00t/data/stats.py --dataset-path $OUT/lerobot/gr1_mv.ColorButtonBox_L0 --embodiment-tag GR1
```

Datasets store the `collect` profile: egoview in "bg crop" (`ego_view`, modality key
`ego_view_bg_crop_pad_res256_freq20`, as the base GR1 checkpoint reads it) and "pad"
(`ego_view_pad`) framings, `robotview`, and the four mounted cameras in "square"
framing. Pixel formats follow Multiview-15views (ego_view yuv420p, others yuv444p).
The language is the string the eval env sends
(`unlocked_waist: press the <color> button on the side of the box`).

`hdf5/ColorButtonBox_L<n>.hdf5` holds, per demo, the scene xml, `ep_meta`, MuJoCo
states and robosuite actions, in the layout of the released GR1 teleop HDF5 files
(`demo_<n>` = LeRobot episode `n - 1`). Re-render demos with:

```bash
$PY multiview/scripts/replay_hdf5.py --hdf5 $OUT/hdf5/ColorButtonBox_L0.hdf5 --demos 1 --out_dir ~/tmp/replay
```

## Notes

- Render GPU: robosuite renders on the EGL device whose index is `CUDA_VISIBLE_DEVICES`,
  and EGL numbers GPUs differently from nvidia-smi on this host (EGL 0..7 = GPU
  3, 2, 0, 1, 7, 6, 4, 5). `collect_scripted.py --gpu` takes the nvidia-smi index and
  converts it (`multiview_tasks/egl_devices.py`). For other sim scripts, set
  `CUDA_VISIBLE_DEVICES=$(python -c "from multiview_tasks.egl_devices import gpu_to_egl_index; print(gpu_to_egl_index()[N])")`.

- `reset(seed=...)` reseeds the robosuite rng, so the same seed gives the same
  scene across levels and profiles. Object sampling also uses the global numpy
  rng, which `RoboCasaEnv.reset` seeds.
- `cam3`/`cam5` need a checkpoint fine-tuned with the extra video keys; the base
  GR1 checkpoint only accepts `cam1`.
