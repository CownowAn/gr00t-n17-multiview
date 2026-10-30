#!/usr/bin/env bash
# N1.7 counterpart of multiview/finetune/eval_trace.sh: evaluate one checkpoint (server on
# GPU <gpu>, sim with the N1.7 sim venv rendering on RENDER_GPU) with per-episode logs.
#   TASK=StopRollingCanTable [DATA_TAG=L3] [N=100] [LEVELS=3] [N_ENVS=5] [MAX_EP_STEPS=200] [RENDER_GPU=<gpu>]
#   eval_trace_n17.sh <cam> <run_tag> <step> <gpu> <port>
set -u
export __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HOME=/nfs/data/sohyun0423/hf_cache
[ -f ~/.cache/huggingface/token ] && export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CAM=$1 TAG=$2 STEP=$3 GPU=$4 PORT=$5 N=${N:-100} LEVELS=${LEVELS:-3}
TASK=${TASK:-StopRollingCanTable}
REPO=$HOME/project/Isaac-GR00T-n1.7mv
# EDGE=1: an EE-VLA run (multiview_gr1_n17_eevla, "_ee" in the run name).
if [ "${EDGE:-0}" = 1 ]; then ROOT=/nfs/data/sohyun/checkpoints/multiview_gr1_n17_eevla/$TASK EE=_ee
else ROOT=/nfs/data/sohyun/checkpoints/multiview_gr1_n17/$TASK EE=; fi
RUN=${TASK}_${DATA_TAG:-L3}_${CAM}${EE}_bs64_lr1e-4_${TAG}
CK=$ROOT/$RUN/$RUN/checkpoint-$STEP
# Runs on another dataset view (DATA_TAG other than L3) get their own eval dirs.
DT=; [ "${DATA_TAG:-L3}" != L3 ] && DT=_${DATA_TAG}
SEED=${EVAL_SEED:-10000}; SX=; [ "$SEED" != 10000 ] && SX=_seed$SEED
OUT=$ROOT/eval_trace/${CAM}${DT}_${TAG}_ckpt${STEP}_n${N}$SX${OUTX:-}
SIM_PY=$REPO/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python
[ -x "$SIM_PY" ] || SIM_PY=$HOME/project/Isaac-GR00T/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/multiview
mkdir -p "$OUT"
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/status.log"; }
[ -d "$CK" ] || { log "missing checkpoint $CK"; exit 1; }
if [ -z "${FORCE:-}" ] && grep -q "all done" "$OUT/status.log" 2>/dev/null; then echo "already done: $OUT"; exit 0; fi
RENDER_GPU=${RENDER_GPU:-$GPU} N_ENVS=${N_ENVS:-5}
EGL=$($SIM_PY -c "from multiview_tasks.egl_devices import gpu_to_egl_index; print(gpu_to_egl_index()[$RENDER_GPU])" 2>/dev/null | tail -1)
log "server on GPU $GPU port $PORT ($CK); rendering on GPU $RENDER_GPU (EGL $EGL), $N_ENVS envs"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path "$CK" \
  --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port "$PORT" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until ss -ltn | grep -q ":$PORT "; do
  kill -0 $SERVER 2>/dev/null || { log "server died"; exit 1; }
  sleep 5
done
log "server up"
for L in $LEVELS; do
  GR1_MV_EPISODE_LOG_DIR=$OUT/episodes_L$L CUDA_VISIBLE_DEVICES=$EGL $SIM_PY gr00t/eval/rollout_policy.py \
    --env-name gr1_mv/${TASK}_L${L}_${CAM} --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
    --n-episodes "$N" --n-envs "$N_ENVS" --n-action-steps 8 --max-episode-steps "${MAX_EP_STEPS:-200}" --seed "$SEED" \
    --video-dir "$OUT/videos_L$L" > "$OUT/rollout_L$L.log" 2>&1
  log "L$L: $(grep -i 'success rate' "$OUT/rollout_L$L.log" | tail -1)"
done
log "all done"
