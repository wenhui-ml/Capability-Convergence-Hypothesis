#!/bin/bash
# Exp E pilot round 6: deep-recurrent-front hybrid + tight-capacity pure state.
cd .
S=./scratch
PY=python
CUDA_VISIBLE_DEVICES=0 $PY expE_composite/train_comp.py --arch e_hybrid_d2a --seed 0 \
  --steps 100000 --answer_w 4 --out "$S/p6_d2a.json" > "$S/p6_d2a.log" 2>&1 &
# wait for GPU 1 (p5 state3) to free, then start the small pure state
( while pgrep -f 'p5_state3_100k' > /dev/null; do sleep 30; done
  CUDA_VISIBLE_DEVICES=1 $PY expE_composite/train_comp.py --arch e_state3_small --seed 0 \
    --steps 100000 --answer_w 4 --out "$S/p6_small.json" > "$S/p6_small.log" 2>&1 ) &
wait
touch "$S/P6_DONE"
