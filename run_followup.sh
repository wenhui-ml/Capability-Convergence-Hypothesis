#!/bin/bash
# Follow-up phase runner (PREREGISTRATION.md amendment 7): GPUs 0-3.
# Heavy training first (C-iii seeds, B-iii 24k, D failing arm), then the cheap
# A1 n=200 inference sweep, then Exp D v2. Two passes so externally-killed
# stragglers are re-queued. Frozen first-phase files are never touched.
cd .
PY=python
LOG=logs/followup.log
echo "=== followup start $(date) pid $$" >> $LOG
mkdir -p results/expB/followup results/expA/followup

for pass in 1 2; do
  awk '{out=$NF; if (out ~ /\.json$/) { if (system("test -f " out) != 0) print } else print }' \
    logs/jobs_followup2.txt > logs/followup_todo.txt
  n=$(wc -l < logs/followup_todo.txt)
  echo "$(date) pass$pass queue: $n jobs" >> $LOG
  if [ "$n" -gt 0 ]; then
    $PY common/launch.py --jobs logs/followup_todo.txt --gpus 0,1,2,3 --per_gpu 4 \
      --log_dir logs/run_followup_p${pass} >> $LOG 2>&1
  fi
done

# Exp D v2 needs the b_hybrid_b1 / 24k checkpoints -> strictly after training
CUDA_VISIBLE_DEVICES=0 $PY expD_comm/d_commensurability_v2.py \
  --out results/expD/d_comm_v2.json >> $LOG 2>&1

echo "=== followup done $(date)" >> $LOG
touch logs/FOLLOWUP_DONE
