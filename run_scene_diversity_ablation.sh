#!/bin/bash
# Scene diversity ablation: v8 with 8/16/32/64/128 unique scenes, 256 total envs
# All runs on cuda:1, 300 iterations each, sequential
#
# Freeze/crash resilient:
#   - Each run is retried up to MAX_RETRIES times
#   - On retry, automatically resumes from the latest saved checkpoint
#   - Watchdog kills the process if no new checkpoint appears within FREEZE_TIMEOUT_S seconds
#   - Logs per run written to logs/scene_diversity_ablation/

DEVICE="cuda:1"
MAX_RETRIES=5
FREEZE_TIMEOUT_S=1800   # 30 min — kill if no new checkpoint in this window
SAVE_INTERVAL=25        # must match runner.save_interval in configs
LOG_DIR="logs/scene_diversity_ablation"
mkdir -p "$LOG_DIR"

# Find the latest model_N.pt in a run dir (returns empty string if none)
latest_checkpoint() {
  local run_dir="$1"
  ls "$run_dir"/model_*.pt 2>/dev/null | sort -V | tail -1
}

# Find the run dir for a given run_name (newest match)
find_run_dir() {
  local run_name="$1"
  ls -dt logs/rsl_rl/scene_factory_goal_reaching_roads/*"${run_name}"* 2>/dev/null | head -1
}

run_with_retry() {
  local N="$1"
  local CONFIG="configs/scene_factory/generated/scene_factory_${N}unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml"
  local RUN_NAME="scene_factory_${N}unique_256total_random_train_fastgoal_v8_sysid4_noweather"
  local LOG_FILE="$LOG_DIR/run_${N}unique.log"
  local attempt=0

  echo "========================================"
  echo "Starting: ${N} unique scenes  (log: $LOG_FILE)"
  echo "========================================"

  while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    echo "[$(date '+%H:%M:%S')] Attempt $attempt / $MAX_RETRIES for ${N}-unique run" | tee -a "$LOG_FILE"

    # Find latest checkpoint to resume from (if any prior attempt saved one)
    local run_dir
    run_dir=$(find_run_dir "$RUN_NAME")
    local resume_arg=""
    if [[ -n "$run_dir" ]]; then
      local ckpt
      ckpt=$(latest_checkpoint "$run_dir")
      if [[ -n "$ckpt" ]]; then
        echo "[$(date '+%H:%M:%S')] Resuming from: $ckpt" | tee -a "$LOG_FILE"
        resume_arg="--resume_from $ckpt"
      fi
    fi

    # Launch training in background so we can watch it
    PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
      --config "$CONFIG" \
      --headless \
      --device "$DEVICE" \
      $resume_arg \
      >> "$LOG_FILE" 2>&1 &
    local PID=$!
    echo "[$(date '+%H:%M:%S')] PID=$PID" | tee -a "$LOG_FILE"

    # Watchdog: poll for new checkpoints; kill if frozen
    local last_ckpt_time=$SECONDS
    local last_ckpt=""
    while kill -0 "$PID" 2>/dev/null; do
      sleep 60
      local run_dir_now
      run_dir_now=$(find_run_dir "$RUN_NAME")
      if [[ -n "$run_dir_now" ]]; then
        local new_ckpt
        new_ckpt=$(latest_checkpoint "$run_dir_now")
        if [[ "$new_ckpt" != "$last_ckpt" && -n "$new_ckpt" ]]; then
          echo "[$(date '+%H:%M:%S')] New checkpoint: $new_ckpt" | tee -a "$LOG_FILE"
          last_ckpt="$new_ckpt"
          last_ckpt_time=$SECONDS
        fi
      fi
      local elapsed=$(( SECONDS - last_ckpt_time ))
      if [[ $elapsed -gt $FREEZE_TIMEOUT_S ]]; then
        echo "[$(date '+%H:%M:%S')] FROZEN: no new checkpoint in ${elapsed}s — killing PID=$PID" | tee -a "$LOG_FILE"
        kill -9 "$PID" 2>/dev/null
        sleep 5
        break
      fi
    done

    wait "$PID" 2>/dev/null
    local EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
      echo "[$(date '+%H:%M:%S')] SUCCESS: ${N}-unique run finished." | tee -a "$LOG_FILE"
      return 0
    else
      echo "[$(date '+%H:%M:%S')] FAILED (exit $EXIT_CODE) — will retry..." | tee -a "$LOG_FILE"
      sleep 10
    fi
  done

  echo "[$(date '+%H:%M:%S')] GAVE UP after $MAX_RETRIES attempts for ${N}-unique run." | tee -a "$LOG_FILE"
  return 1
}

# Run all 5 sequentially
for N in 8 16 32 64 128; do
  run_with_retry "$N" || echo "WARNING: ${N}-unique run did not complete successfully, continuing..."
done

echo "All scene diversity ablation runs complete."
