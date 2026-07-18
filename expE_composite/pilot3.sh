#!/bin/bash
# Exp E pilot round 3: weight tying, 24k steps, three configs in parallel.
cd .
S=./scratch
PY=python
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_state --seed 0 \
  --steps 24000 --out "$S/p3_state.json" > "$S/p3_state.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_hybrid --seed 0 \
  --steps 24000 --out "$S/p3_hyb.json" > "$S/p3_hyb.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY expE_composite/train_comp.py --arch e_hybrid_big --seed 0 \
  --steps 24000 --out "$S/p3_hybbig.json" > "$S/p3_hybbig.log" 2>&1 &
wait
touch "$S/P3_DONE"
