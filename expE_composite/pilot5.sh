#!/bin/bash
# Exp E pilot round 5: unconfounded long-budget test — round-2's known-good
# supervision config (answer_w=4, untied) at 100k steps.
cd .
S=./scratch
PY=python
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_hybrid_big --seed 0 \
  --steps 100000 --answer_w 4 --out "$S/p5_hybbig100k.json" > "$S/p5_hybbig100k.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_state3 --seed 0 \
  --steps 100000 --answer_w 4 --out "$S/p5_state3_100k.json" > "$S/p5_state3_100k.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY expE_composite/train_comp.py --arch e_hybrid_big --seed 1 \
  --steps 100000 --answer_w 4 --out "$S/p5_hybbig100k_s1.json" > "$S/p5_hybbig100k_s1.log" 2>&1 &
wait
touch "$S/P5_DONE"
