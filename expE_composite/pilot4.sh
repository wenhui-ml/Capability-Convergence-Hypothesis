#!/bin/bash
# Exp E pilot round 4 (untied): time hypothesis vs depth hypothesis.
cd .
S=./scratch
PY=python
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_hybrid_big --seed 0 \
  --steps 60000 --answer_w 8 --out "$S/p4_hybbig60k.json" > "$S/p4_hybbig60k.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_hybrid3 --seed 0 \
  --steps 30000 --answer_w 8 --out "$S/p4_hyb3.json" > "$S/p4_hyb3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY expE_composite/train_comp.py --arch e_state3 --seed 0 \
  --steps 60000 --answer_w 8 --out "$S/p4_state3.json" > "$S/p4_state3.log" 2>&1 &
wait
touch "$S/P4_DONE"
