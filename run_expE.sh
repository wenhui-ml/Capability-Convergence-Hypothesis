#!/bin/bash
# One-command reproduction of Experiment E (the composite conjunction witness).
#
# Trains the 5 frozen arms x 3 seeds (PREREGISTRATION.md amendment 9b: 120k
# steps, answer_w=4, untied), then runs the analysis and regenerates the figure.
# Self-contained: no external job manifest, no fixed GPU layout. Already-finished
# outputs are skipped, so the script is safe to re-run after an interruption.
#
# Requires a CUDA GPU (train_comp.py trains on cuda). Full run is ~15 training
# jobs of 120k steps each (hours on a single GPU). To pick a GPU:
#   CUDA_VISIBLE_DEVICES=0 bash run_expE.sh
#
# Note: to only regenerate the figure from the result JSONs already shipped in
# this repo (no training, no GPU), run instead:
#   python analyze_expE.py && python expE_composite/plot_e.py
set -euo pipefail
cd "$(dirname "$0")"
PY=python
mkdir -p results/expE

ARMS="e_hybrid_d2a e_state3 e_attn8 e_hybrid_b1_d2a e_hybrid_local_d2a"
SEEDS="0 1 2"

for arch in $ARMS; do
  for s in $SEEDS; do
    out="results/expE/${arch}_s${s}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out already exists"
      continue
    fi
    echo "[train] $arch seed $s -> $out"
    $PY expE_composite/train_comp.py --arch "$arch" --seed "$s" \
      --steps 120000 --answer_w 4 --out "$out"
  done
done

# D-ii (amendment 10): channel complementarity on the solving-hybrid finals.
$PY expE_composite/d2_complementarity.py \
  --arch e_hybrid_d2a --ckpts 'results/expE/e_hybrid_d2a_s*_ckpt_final.pt' \
  --out results/expE/d2_complementarity.json

# Analysis + figure (CPU-only).
$PY analyze_expE.py
$PY expE_composite/plot_e.py
echo "=== Exp E done: results/expE/summary_expE.json, figs/expE_witness.{pdf,png}"
