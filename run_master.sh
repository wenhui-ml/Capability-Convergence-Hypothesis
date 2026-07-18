#!/bin/bash
# Self-sufficient pipeline runner (v2): ONE combined priority queue so all
# 16 workers (GPUs 4-7 x 4) stay busy end-to-end. Survives disconnects (setsid).
cd .
PY=python
LOG=logs/master.log
echo "=== master v2 start $(date) pid $$" >> $LOG

# let any stray session-tied jobs drain first (avoids double-running in-flight)
while pgrep -f "train_s5\.py|train_floor\.py|a1_capability\.py|a2_alignment\.py" >/dev/null; do
  sleep 120
done

# two passes: pass 2 catches externally-killed stragglers
for pass in 1 2; do
  # priority order: starred-prediction cells, main B/C grid, Exp A, 100k grid
  cat logs/jobs_critical.txt logs/jobs_BC_v2.txt logs/jobs_A.txt \
      logs/jobs_phase2_rest.txt \
    | awk '{out=$NF; if (out ~ /\.json$/) { if (system("test -f " out) != 0) print } else print }' \
    > logs/combined_todo.txt
  n=$(wc -l < logs/combined_todo.txt)
  echo "$(date) pass$pass combined queue: $n jobs" >> $LOG
  if [ "$n" -gt 0 ]; then
    $PY common/launch.py --jobs logs/combined_todo.txt --gpus 4,5,6,7 --per_gpu 4 \
      --log_dir logs/run_master_v2_p${pass} >> $LOG 2>&1
  fi
done

# Exp D needs the trained hybrid checkpoints -> strictly after all training
CUDA_VISIBLE_DEVICES=4 $PY expD_comm/d_commensurability.py \
  --out results/expD/d_comm.json >> $LOG 2>&1

# A2 alignment stage (all features on disk by now)
CUDA_VISIBLE_DEVICES=4 $PY expA_dissociation/a2_alignment.py --stage align >> $LOG 2>&1

# figures + pre-registered verdicts (defensive to missing cells)
CUDA_VISIBLE_DEVICES=4 $PY expC_s5/plot_c.py >> $LOG 2>&1
CUDA_VISIBLE_DEVICES=4 $PY expB_floor/plot_b.py >> $LOG 2>&1
CUDA_VISIBLE_DEVICES=4 $PY expA_dissociation/plot_a.py >> $LOG 2>&1
echo "=== master v2 done $(date)" >> $LOG
touch logs/MASTER_DONE
