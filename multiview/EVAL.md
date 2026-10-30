# Evaluating GR00T N1.7 on the six GR1 multi-view tasks

How we evaluate GR00T N1.7 policies on the six multi-view tasks (StopRollingCanTable,
StopRollingBallTable, ColorButtonBoxV3, HoldDoorTakeBlockV3, HoldDoorTakeFruitV3, PushDrawerBoxV3):
where the task and camera code lives, the exact settings (episodes, seeds, max steps), and how we break
the results down. The datasets are described in the dataset README shared with them.

## Where things are

| path | what it is |
|---|---|
| `multiview/multiview_tasks/tasks/` | Task env code, one file per task: `roll_can_table.py`, `roll_ball_table.py`, `color_button.py` (ColorButtonBoxV3), `door_hold.py` (HoldDoorTakeBlockV3), `door_fruit.py`, `push_drawer.py` (other files are earlier tasks we no longer use) |
| `multiview/multiview_tasks/cameras.py` | Every mounted camera (parent body, pose, fovy) and the camera profiles cam1 / cam3* / cam5* |
| `multiview/multiview_tasks/registry.py` | `MultiViewGrootEnv` (camera profile, per-episode seeding, episode logging) and the gym registration `gr1_mv/<Task>_L<level>_<profile>` |
| `multiview/finetune_n17/gr1_mv_modality.py`, `modality_<profile>.py` | The N1.7 modality configs we trained with (video keys per profile, 8-step action chunks) |
| `multiview/finetune_n17/eval_trace_n17.sh` | Our eval script (starts the policy server, runs the rollout, writes per-episode logs) |
| `gr00t/eval/rollout_policy.py`, `gr00t/eval/sim/env_utils.py`, `gr00t/eval/sim/wrapper/video_recording_wrapper.py` | Small changes to the upstream N1.7 files: the `gr1_mv/` env prefix, per-episode seeds, per-episode info |
| `multiview/analysis/` | Result breakdown scripts (see below) |

## Setup

1. Rollouts run in the robocasa-gr1-tabletop-tasks sim venv (`gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv`);
   the policy server runs in the main N1.7 venv.
2. Put the task package on the path for the rollout: `export PYTHONPATH=$REPO:$REPO/multiview`
   (or `pip install -e multiview` into the sim venv).
3. HoldDoorTakeFruitV3 uses RoboCasa's objaverse objects (apple, orange, lemon, lime, peach, potato),
   which a standard robocasa-gr1-tabletop-tasks install includes.
4. Rendering uses EGL (`MUJOCO_GL=egl`). The EGL device order can differ from `nvidia-smi` order:
   `multiview_tasks.egl_devices.gpu_to_egl_index()` maps a GPU index to its EGL index, and we start the
   rollout with `CUDA_VISIBLE_DEVICES=<EGL index>` (see `eval_trace_n17.sh`).
5. Our scripts contain paths from our servers (`/nfs/data/sohyun/...`, `$HOME/project/...`); change them
   to yours.

## Env IDs and camera profiles

All evaluation is at level 3 (the level of the training data): `gr1_mv/<Task>_L3_<profile>`.

| task | cam1 | cam3 | cam5 |
|---|---|---|---|
| StopRollingCanTable | `cam1` | `cam3r` | `cam5r` |
| StopRollingBallTable | `cam1` | `cam3r` | `cam5r` |
| ColorButtonBoxV3 | `cam1` | `cam3b` | `cam5b` |
| HoldDoorTakeBlockV3 | `cam1` | `cam3` | `cam5d` |
| HoldDoorTakeFruitV3 | `cam1` | `cam3` | `cam5d` |
| PushDrawerBoxV3 | `cam1` | `cam3` | `cam5w` |

The env returns each camera under the same key as the dataset's modality key
(`video.ego_view_bg_crop_pad_res256_freq20`, `video.roll_diag_left`, ...). **If your model uses other
camera key names (e.g. generic `view1`..`view4` for multi-task training), rename the env's observation
keys before they reach the policy**, for example in a small wrapper around the policy server.

## Evaluation settings

| setting | value |
|---|---|
| episodes per checkpoint | 100 (`--n-episodes 100`), level 3 only |
| parallel envs | 5 (`--n-envs 5`) |
| action steps executed per policy call | 8 (`--n-action-steps 8`) |
| max steps per episode (20 Hz) | StopRollingCan / StopRollingBall **200**; ColorButtonBoxV3 / PushDrawerBoxV3 **400**; HoldDoorTakeBlockV3 / HoldDoorTakeFruitV3 **500** |
| seed | `--seed 10000`. Episode k of vector env i uses seed 10000 + i + 5k, so every checkpoint and model sees the same scenes |
| policy server | `gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port <port>` (our models were trained with NEW_EMBODIMENT) |
| per-episode logs | `GR1_MV_EPISODE_LOG_DIR=<dir>` writes one jsonl record per episode (seed, target, success, per-step object and hand positions); `multiview/analysis/` reads these |

What `eval_trace_n17.sh` runs, for one checkpoint:

```bash
# policy server (main N1.7 venv)
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path <ckpt> \
  --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port 6000 &
# rollout (sim venv)
GR1_MV_EPISODE_LOG_DIR=<out>/episodes_L3 CUDA_VISIBLE_DEVICES=<EGL index> PYTHONPATH=$REPO:$REPO/multiview \
  <sim venv python> gr00t/eval/rollout_policy.py \
  --env-name gr1_mv/StopRollingCanTable_L3_cam5r --policy-client-host 127.0.0.1 --policy-client-port 6000 \
  --n-episodes 100 --n-envs 5 --n-action-steps 8 --max-episode-steps 200 --seed 10000 --video-dir <out>/videos_L3
```

Notes from our runs:

- Compare models on the **same seed set**. The target and the roll direction depend on the seed; on the
  can task a different seed set moved one of our models from 36% to 48% because it changed the mix of
  roll directions.
- Repeated evaluations of the same checkpoint on the same seeds varied by about ±3 points (the policy
  samples its action chunks from noise).
- When several evaluations share a machine, an episode's video writer (ffmpeg) or the policy server can
  time out; the rollout then stops without a result. Re-running that checkpoint fixes it.

## Success criteria (from the task code)

| task | success |
|---|---|
| StopRollingCanTable / StopRollingBallTable | the object was pushed, no hand moved before the push (a "false start"), and the object stays on the table, on its own side of the midline, and stops moving along its path |
| ColorButtonBoxV3 | the target button is pressed and no other button is pressed |
| HoldDoorTakeBlockV3 / HoldDoorTakeFruitV3 | the target item is out on the work surface, the other items are still in the cabinet, and the door is still open |
| PushDrawerBoxV3 | the target drawer is pushed in most of the way and no other drawer is pushed in beyond a small threshold |

## Breakdown scripts

- `python multiview/analysis/rolling_breakdown.py <eval dir>`: can / ball success per roll direction
  (front / diag / side), number of false starts, and when the blocking hand first moved relative to the push.
- `python multiview/analysis/door_first_touch.py <eval dir>`: door tasks: success and which item was
  touched first (target / wrong / none), separating "picked the wrong item" from "did not reach".
