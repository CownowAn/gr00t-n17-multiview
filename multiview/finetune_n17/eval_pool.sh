#!/usr/bin/env bash
# Evaluate a list of checkpoints with one worker per GPU pulling jobs from a shared queue.
#   eval_pool.sh <jobs_file> <wait> <gpu>...
# jobs_file: one "<cam> <run_tag> <step>" per line (consumed as jobs start).
# wait: "none" or "file:<path>:<text>" (start once that file contains <text>).
# A worker only takes a job while no other user has a process on its GPU (unless
# IGNORE_OTHERS is set, when the user has cleared sharing the GPU). A GPU listed
# twice gets two workers. EVAL_SCRIPT (default eval_run_ckpt.sh) runs each job, on port
# PORT_BASE (default 5580) + worker index.
set -u
JOBS=$1 WAIT=$2; shift 2
DIR=$(cd "$(dirname "$0")" && pwd)
LOCK=$JOBS.lock
LOG=$JOBS.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [[ "$WAIT" == file:* ]]; then
  rest=${WAIT#file:}; f=${rest%%:*}; text=${rest#*:}
  log "waiting for '$text' in $f"
  until grep -q "$text" "$f" 2>/dev/null; do sleep 60; done
fi

others_on() {  # number of other users' compute processes on GPU $1
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$1" '$1==g {print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}' |
    while read -r pid; do ps -o user= -p "$pid"; done | grep -vc "^$(whoami)$"
}

worker() {
  local gpu=$1 port=$2
  while :; do
    [ -n "${IGNORE_OTHERS:-}" ] || while [ "$(others_on "$gpu")" -gt 0 ]; do sleep 120; done
    local job
    job=$(flock "$LOCK" bash -c 'head -n1 "$0"; sed -i 1d "$0"' "$JOBS")
    [ -z "$job" ] && break
    log "GPU $gpu: $job"
    # shellcheck disable=SC2086
    "$DIR/${EVAL_SCRIPT:-eval_run_ckpt.sh}" $job "$gpu" "$port"
  done
  log "GPU $gpu: no more jobs"
}

i=0
for gpu in "$@"; do
  worker "$gpu" $((${PORT_BASE:-5580} + i)) &
  i=$((i + 1))
done
wait
log "all done"
