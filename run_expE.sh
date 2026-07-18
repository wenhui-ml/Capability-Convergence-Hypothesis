#!/bin/bash
# Exp E formal runs (amendment 9b): 5 frozen arms x 3 seeds x 120k steps.
# Two passes for externally-killed stragglers; skip finished outputs.
cd .
PY=python
LOG=logs/expE.log
echo "=== expE final start $(date) pid $$" >> $LOG
mkdir -p results/expE

for pass in 1 2; do
  awk '{out=$NF; if (out ~ /\.json$/) { if (system("test -f " out) != 0) print } else print }' \
    logs/jobs_expE_final.txt > logs/expE_todo.txt
  n=$(wc -l < logs/expE_todo.txt)
  echo "$(date) pass$pass queue: $n jobs" >> $LOG
  if [ "$n" -gt 0 ]; then
    $PY common/launch.py --jobs logs/expE_todo.txt --gpus 0,1,2,3 --per_gpu 3 \
      --log_dir logs/run_expE_p${pass} >> $LOG 2>&1
  fi
done

# D-ii (amendment 10) on the solving hybrid finals, strictly after training
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/d2_complementarity.py \
  --arch e_hybrid_d2a --ckpts 'results/expE/e_hybrid_d2a_s*_ckpt_final.pt' \
  --out results/expE/d2_complementarity.json >> $LOG 2>&1

echo "=== expE final done $(date)" >> $LOG
touch logs/EXPE_DONE
