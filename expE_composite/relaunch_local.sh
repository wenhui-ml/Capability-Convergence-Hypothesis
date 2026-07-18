#!/bin/bash
# Rebalance: the three e_hybrid_local_d2a jobs (killed at 12k steps) rerun on
# idle GPUs 0/1; attn8 keeps GPUs 2/3 with reduced contention.
cd .
PY=python
L=logs/run_expE_p1
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_hybrid_local_d2a --seed 0 \
  --steps 120000 --answer_w 4 --out results/expE/e_hybrid_local_d2a_s0.json > $L/job012_moved.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_hybrid_local_d2a --seed 1 \
  --steps 120000 --answer_w 4 --out results/expE/e_hybrid_local_d2a_s1.json > $L/job013_moved.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_hybrid_local_d2a --seed 2 \
  --steps 120000 --answer_w 4 --out results/expE/e_hybrid_local_d2a_s2.json > $L/job014_moved.log 2>&1 &
wait
