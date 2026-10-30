#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on one multiview GR1 task/profile (NEW_EMBODIMENT, 8-step actions).
#   [TASK=StopRollingCanTable] [DATA_TAG=L3] [MAX_STEPS=10000] [SAVE_STEPS=1000] [RUN_SUFFIX=]
#   [COLOR_JITTER="brightness 0.3 contrast 0.4 saturation 0.2 hue 0.0"]
#   multiview/finetune_n17/run_finetune_n17.sh <profile> <gpus, e.g. 6,7> <master_port>
# The dataset is the N1.7 view of the N1.6 LeRobot dataset (own meta/ with N1.7 stats,
# data/ and videos/ symlinked). Run from the N1.7 repo root.
set -euo pipefail
PROFILE=$1 GPUS=$2 PORT=${3:-29600}
TASK=${TASK:-StopRollingCanTable} DATA_TAG=${DATA_TAG:-L3}
MAX_STEPS=${MAX_STEPS:-10000} SAVE_STEPS=${SAVE_STEPS:-1000}
DATA=/nfs/data/sohyun/data/multiview_gr1_n17/$TASK/gr1_mv.${TASK}_${DATA_TAG}
# EDGE=1: EE-VLA (ego view -> VLM, other views -> per-view DINOv3), EDGE_DROPOUT_P (default 0.2).
EDGE=${EDGE:-0} EDGE_TAG= ROOT=/nfs/data/sohyun/checkpoints/multiview_gr1_n17 EDGE_ARGS=()
if [ "$EDGE" = 1 ]; then
  EDGE_TAG=_ee ROOT=/nfs/data/sohyun/checkpoints/multiview_gr1_n17_eevla
  EDGE_ARGS=(-- --use_edge_experts --edge_dropout_p "${EDGE_DROPOUT_P:-0.2}")
  # EDGE_DROPOUT_UNIT=view: drop each edge view independently instead of all of a sample's.
  [ -n "${EDGE_DROPOUT_UNIT:-}" ] && EDGE_ARGS+=(--edge_dropout_unit "$EDGE_DROPOUT_UNIT")
fi
RUN=${TASK}_${DATA_TAG}_${PROFILE}${EDGE_TAG}_bs64_lr1e-4_$((MAX_STEPS / 1000))k${RUN_SUFFIX:-}
OUT=$ROOT/$TASK/$RUN
mkdir -p "$OUT/logs"

export PATH=$PWD/.venv/bin:$PATH HF_HOME=/nfs/data/sohyun0423/hf_cache
# Cosmos-Reason2-2B is gated: use this host's own token (HF_HOME moves the default token path).
[ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export CUDA_VISIBLE_DEVICES=$GPUS NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
export GLOBAL_BATCH_SIZE=64 MAX_STEPS SAVE_STEPS MASTER_PORT=$PORT USE_WANDB=${USE_WANDB:-1} DATALOADER_NUM_WORKERS=8
# wandb: same project as the N1.6 runs, grouped per task and model version.
export WANDB_RUN_GROUP=${TASK}_n17
[ -n "${COLOR_JITTER:-}" ] && export COLOR_JITTER_PARAMS=$COLOR_JITTER

SAVE_LIMIT=${SAVE_LIMIT:-10} bash multiview/finetune_n17/finetune_n17.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path multiview/finetune_n17/modality_${PROFILE}.py \
  --output-dir "$OUT" \
  --experiment-name "$RUN" \
  --save-only-model \
  --wandb-project "${WANDB_PROJECT:-gr00t-multiview}" \
  "${EDGE_ARGS[@]}" \
  2>&1 | tee "$OUT/logs/train.log"
