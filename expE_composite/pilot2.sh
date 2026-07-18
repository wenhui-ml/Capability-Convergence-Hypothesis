#!/bin/bash
# Exp E pilot round 2: running-answer supervision, three arch configs in parallel.
cd .
S=./scratch
PY=python
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_state --seed 0 \
  --steps 21000 --out "$S/p2_state.json" > "$S/p2_state.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_hybrid --seed 0 \
  --steps 21000 --out "$S/p2_hyb.json" > "$S/p2_hyb.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY expE_composite/train_comp.py --arch e_hybrid_big --seed 0 \
  --steps 21000 --out "$S/p2_hybbig.json" > "$S/p2_hybbig.log" 2>&1 &
wait
for f in p2_state p2_hyb p2_hybbig; do echo "=== $f"; tail -2 "$S/$f.log"; done
