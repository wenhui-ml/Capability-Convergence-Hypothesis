# Capability-Convergence-Hypothesis

Official repository for "Capability from Access Structure, Not Scale: Lower
Bounds and Pre-Registered Tests for Hybrid Sequence Models" (arXiv:2607.14144).

This repository is the `experiments/` directory referenced throughout the paper:
paths the paper writes as `experiments/census_frontier.py`,
`experiments/PREREGISTRATION.md`, `experiments/collision_naturalness/`, etc.,
correspond to the same files at the root here.

## Contents

- `PREREGISTRATION.md` — the pre-registration protocol (frozen 2026-07-10) and its amendments.
- `REPORT.md` — full results write-up and per-prediction scorecard.
- `common/` — shared model, task, and launcher code.
- `expA_dissociation/` — Exp A: representational convergence vs. capability dissociation (Pythia/Mamba/RWKV).
- `expB_floor/` — Exp B (P1): information floor / scissors gap.
- `expC_s5/` — Exp C (P2): S5 state-tracking bifurcation.
- `expD_comm/` — Exp D (P3/P3'): channel commensurability / complementarity.
- `expE_composite/` — Exp E: composite conjunction witness.
- `collision_naturalness/` — Appendix G.6: write-time key-separability corpus measurement.
- `census_frontier.py` — Appendix D: frontier-subset rule for the architecture census.
- `results/` — per-experiment result summaries (JSON) behind the paper's figures.
- `figs/` — the paper's measured figures (PDF/PNG).

## Reproducing the figures

Trained-model checkpoints (`*.pt`) are not included to keep the repository light;
the result JSONs under `results/` are sufficient to regenerate every figure via
the per-experiment `plot_*.py` scripts (e.g. `expB_floor/plot_b.py`,
`expC_s5/plot_c.py`, `expA_dissociation/plot_a.py`, `expE_composite/plot_e.py`).
