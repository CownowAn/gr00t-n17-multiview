#!/usr/bin/env bash
# N1.7 training slot (counterpart of multiview/finetune/run_task_slot.sh): pulls "<profile> <hue0|default>"
# lines from a shared queue, fine-tunes 10k steps (save every 1k, weights only), queues that
# run's 8 checkpoint evaluations on this slot (server on the first GPU, rendering on the
# second) and takes the next run. When the queue is empty, runs AFTER (if set) with exec.
#   TASK=StopRollingCanTable [AFTER="cmd"] run_slot_n17.sh <queue> <gpus> <master_port> <eval_port>
set -u
Q=$1 GPUS=$2 PORT=$3 EPORT=$4
export TASK=${TASK:-StopRollingCanTable} DATA_TAG=L3 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_LIMIT=10
REPO=$HOME/project/Isaac-GR00T-n1.7mv
CKPT=/nfs/data/sohyun/checkpoints/multiview_gr1_n17/$TASK
QL=$CKPT/eval_trace/queue_logs
mkdir -p "$CKPT/logs" "$QL"
LOG=$CKPT/logs/train.log
G1=${GPUS%,*} G2=${GPUS#*,}
log() { echo "[$(date '+%F %T')] $(hostname -s): $*" >> "$LOG"; }
free_enough() {
  local g u
  for g in $G1 $G2; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    [ "$u" -lt 12000 ] || return 1
  done
}
cd "$REPO"
while :; do
  until free_enough; do sleep 120; done
  job=$(flock "$Q.lock" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$Q")
  [ -z "$job" ] && break
  read -r PROFILE VARIANT <<< "$job"
  if [ "$VARIANT" = hue0 ]; then
    export COLOR_JITTER="brightness 0.3 contrast 0.4 saturation 0.2 hue 0.0" RUN_SUFFIX=_hue0
  else
    unset COLOR_JITTER; export RUN_SUFFIX=
  fi
  RUN=${TASK}_${DATA_TAG}_${PROFILE}_bs64_lr1e-4_10k${RUN_SUFFIX}
  log "$RUN start on GPU $GPUS"
  multiview/finetune_n17/run_finetune_n17.sh "$PROFILE" "$GPUS" "$PORT" > /dev/null 2>&1
  status=$?
  rm -rf "$CKPT/$RUN/$RUN"/checkpoint-{7000,9000}
  log "$RUN exit $status ($(du -sh "$CKPT/$RUN" | cut -f1))"
  # A failed training stops this slot (no evaluations, no AFTER), so one error does not
  # cascade through the queue.
  [ "$status" = 0 ] || { log "slot $GPUS: stopping after failed $RUN"; exit 1; }
  EJ=$QL/jobs_${PROFILE}_${VARIANT}.txt
  for s in 10000 5000 2000 1000 3000 4000 6000 8000; do echo "$PROFILE 10k${RUN_SUFFIX} $s"; done > "$EJ"
  (cd $REPO/multiview/finetune_n17 && TASK=$TASK DATA_TAG=L3 LEVELS=3 N=100 \
     MAX_EP_STEPS=200 N_ENVS=5 RENDER_GPU=$G2 IGNORE_OTHERS=1 EVAL_SCRIPT=eval_trace_n17.sh \
     PORT_BASE=$EPORT setsid nohup ./eval_pool.sh "$EJ" none "$G1" > /dev/null 2>&1 < /dev/null &)
  log "$RUN evaluations queued on GPU $G1 (render $G2), port $EPORT: $EJ"
  EPORT=$((EPORT + 1))
done
log "slot $GPUS: queue empty"
[ -n "${AFTER:-}" ] && exec bash -c "$AFTER"
