"""Exp A (dissociation), A-iv: partial correlation controlling for scale.

The paper reports that the positive association between cross-family representational
alignment and the absolute retrieval gap is NOT a scale artifact: controlling for the pair's
log geometric-mean parameter count, the partial Spearman correlation stays positive
(+0.30, p=0.009). This script re-derives that number from the shipped results so the claim is
auditable (it was previously computed only in an ephemeral working directory).

Method (identical pair set to expA_dissociation/plot_a.py, panel (d)):
  - cross-family pairs only, at L=896, cell N16_d0.1;
  - x = mutual k-NN alignment (results/expA/a2_alignment.json);
  - y = |retrieval-accuracy gap| between the two models in the pair;
  - z = log of the geometric-mean parameter count of the pair (the scale control);
  - partial Spearman = Pearson correlation of the rank-residuals of x and y after
    (linearly) regressing each on rank(z).

Usage:  python expA_partial_correlation.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "expA_dissociation"))
from plot_a import PARAMS, family, load_a1  # noqa: E402  (paper's own helpers)

ALIGN = os.path.join(HERE, "results", "expA", "a2_alignment.json")
LENGTH = 896
CELL = "N16_d0.1"


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling z: correlate the residuals of the
    rank-transformed variables after regressing rank(x),rank(y) on rank(z)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)

    r, p = stats.pearsonr(resid(rx, rz), resid(ry, rz))
    return r, p


def main():
    aln = json.load(open(ALIGN))["pairs"]
    a1 = load_a1(LENGTH)
    x, y, z = [], [], []
    for key, v in aln.items():
        a, b = key.split("__")
        if family(a) == family(b):          # cross-family only
            continue
        if a not in a1 or b not in a1:
            continue
        gap = abs(a1[a][CELL]["acc"] - a1[b][CELL]["acc"])
        x.append(v["mutual_knn"])
        y.append(gap)
        z.append(np.log(PARAMS[a] * PARAMS[b]) / 2.0)   # log geometric-mean params
    x, y, z = map(np.asarray, (x, y, z))

    r0, p0 = stats.spearmanr(x, y)
    rp, pp = partial_spearman(x, y, z)
    print(f"n cross-family pairs (L={LENGTH}, cell {CELL}): {len(x)}")
    print(f"zero-order Spearman(alignment, |gap|)        : {r0:+.3f}  (p={p0:.2e})")
    print(f"partial Spearman | log geo-mean params        : {rp:+.3f}  (p={pp:.3f})")
    print("paper claims: zero-order +0.48 (p<1e-4); partial +0.30 (p=0.009)")


if __name__ == "__main__":
    main()
