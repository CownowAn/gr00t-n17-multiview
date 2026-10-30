#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the ColosseumV2 dual-arm Panda LeRobot dataset (NEW_EMBODIMENT).
#   [DATA=<lerobot dir>] [MAX_STEPS=60000] [SAVE_STEPS=10000] [EDGE=0|1] [EDGE_DROPOUT_P=0.2] [RUN_SUFFIX=]
#   multiview/colosseum/run_cv2_n17.sh <profile: ext1|ext2|lhand|rhand|ext12|all4> <gpus> <master_port>
# Run from the N1.7 repo root. EDGE=1: the first camera of the profile goes to the VLM, the others to
# per-view DINOv3 encoders (EE-VLA, sample-level edge dropout).
set -euo pipefail
PROFILE=$1 GPUS=$2 PORT=${3:-29650}
DATA=${DATA:-/nfs/data/sohyun/data/colosseum_v2_lerobot/cv2_dual10}
MAX_STEPS=${MAX_STEPS:-60000} SAVE_STEPS=${SAVE_STEPS:-10000}
EDGE=${EDGE:-0} EDGE_TAG= EDGE_ARGS=()
if [ "$EDGE" = 1 ]; then EDGE_TAG=_ee; EDGE_ARGS=(-- --use_edge_experts --edge_dropout_p "${EDGE_DROPOUT_P:-0.2}"); fi
RUN=cv2dual10_${PROFILE}${EDGE_TAG}_bs64_lr1e-4_$((MAX_STEPS / 1000))k${RUN_SUFFIX:-}
OUT=/nfs/data/sohyun/checkpoints/colosseum_v2_n17/$RUN
mkdir -p "$OUT/logs"
export PATH=$PWD/.venv/bin:$PATH HF_HOME=/nfs/data/sohyun0423/hf_cache
[ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=$GPUS NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
export GLOBAL_BATCH_SIZE=64 MAX_STEPS SAVE_STEPS MASTER_PORT=$PORT USE_WANDB=${USE_WANDB:-1} DATALOADER_NUM_WORKERS=8
export WANDB_RUN_GROUP=colosseum_v2_n17 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COLOR_JITTER_PARAMS="${COLOR_JITTER:-brightness 0.3 contrast 0.4 saturation 0.2 hue 0.0}"
SAVE_LIMIT=${SAVE_LIMIT:-6} bash multiview/finetune_n17/finetune_n17.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path multiview/colosseum/cv2_modality_${PROFILE}.py \
  --output-dir "$OUT" \
  --experiment-name "$RUN" \
  --save-only-model \
  --wandb-project "${WANDB_PROJECT:-gr00t-multiview}" \
  "${EDGE_ARGS[@]}" \
  2>&1 | tee "$OUT/logs/train.log"
