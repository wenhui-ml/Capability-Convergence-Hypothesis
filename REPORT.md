# CCH Experiments — Report

Status: **COMPLETE** (all pipeline runs and the analysis/verdicts below finished
2026-07-10).
Follow-up phase (amendment 7): **COMPLETE** — see [Follow-up phase results](#follow-up-phase-amendment-7--results).
Note: `summary_expC.json` and `figs/expC_bifurcation.*` now include the amendment-7a
seeds (c_dp4 general: 8 seeds), under which C-iii fails the frozen ≥2/3 criterion;
the 3-seed primary verdict is preserved in the table below.
Protocol and decision criteria: see [PREREGISTRATION.md](PREREGISTRATION.md) —
criteria frozen before data, amendments logged there with timestamps.

**Scorecard (details below): 11 SUPPORTED, 7 PARTIAL, 1 FAILED** of 19
pre-registered predictions. The failure (D-i, channel commensurability) is
reported as a failure of CCH's prediction P3 as operationalized, per the
reporting commitments.

## What these experiments are for

The paper (CCH_arxiv.tex) is a hypothesis paper whose falsifiable content lives in
P1–P3, but all five prediction figures are labeled "Schematic prediction, not
measured data" — the peer review identified this as the single decisive weakness
(the discriminating evidence is exactly the part not run), and the tension with
Reasoning-Primitives can only be answered by a no-CoT fixed-budget separation.
These experiments turn the schematic figures into measured ones:

| Gap | Experiment | Paper artifact it replaces |
|---|---|---|
| P1 information floor / scissors gap | **B** | Fig. 3 (fig3_fano_floor) |
| P2 state-tracking bifurcation | **C** | Fig. 8 (table2_nh_grid) |
| P3 channel commensurability | **D** (miniature) | Fig. 4 (fig4_commensurability) |
| No real-weights analysis (PRH→CCH bridge) | **A** | new Figure (dissociation) |

## Methods summary

**Common regime** (= paper's R̄): fixed depth, no chain-of-thought, no tools,
fixed per-token compute; only the sequence mixer differs between architectures;
identical data, optimizer, steps within each comparison.

**Exp C** — S5 word problem, 120-way classification of the running product at every
position. Train T=40; evaluate last-position accuracy at T∈{40,64,128,256,512}.
Architectures: DeltaNet β∈(0,1) vs β∈(0,2) (eigenvalue range = reflection unlocked),
DeltaProduct n_h∈{2,4}, diagonal SSM, Transformer (2L and depth-advantaged 8L), LSTM.
2 layers, d_model=128, 3 seeds. Implementation: exact chunked WY delta-rule scan
(validated ≡ sequential scan to 1e-16, incl. gradients).

**Exp B** — executable NA(N,b,B;κ): N composite-key bindings (cluster token + id
token → value token, b=8 bits), semantic-overlap distractors (κ high = noise keys
from planted clusters), single query at the end, L=512. Train on mixed
N∈{2..128}, LM loss over stream + answer loss; evaluate answer-position exact error
and log-loss per N. Pure state m∈{64,256,1024} total recurrent scalars; sliding
window w=32 (receptive field 2·32+conv ≪ binding distance); hybrid = same m=64
state + ONE global-attention layer; hybrid-local = same state + window-32 layer
(NOT access-complete); full attention. Compressibility axis: values shared in
blocks c∈{1,4,16} (effective entropy b/c). 3 seeds.

**Exp A** — Pythia (attention) / Mamba (SSM) / RWKV-4 (linear RNN), all pretrained
on The Pile, 70m–3b, all sharing the GPT-NeoX tokenizer → identical token streams.
A1: N facts ("the secret code of {adj noun} is {value}") planted at depth
d∈{0.1,0.5,0.9} in Pile filler, L∈{896,1900}, exact-match top-1 + log-prob on the
value token, 100 instances/cell. A2: PRH's mutual-kNN (k=10) + CKA between all
model pairs on 1024 shared Pile snippets (mean-pooled per-layer features, max over
layer pairs). A3: alignment vs |capability gap| scatter.

**Exp D** — mutual-kNN alignment between the state-channel layer output and the
attention-layer output of Exp B hybrids at matched positions: final (task-solving)
vs step-400 (failing) vs random-init checkpoints.

## Results

All numbers are means over 3 seeds unless noted; per-seed values are given where
seeds disagree. Verdicts follow the frozen criteria in PREREGISTRATION.md
(including amendments). Raw JSONs: `results/exp{A,B,C,D}/`. Figures:
`figs/expB_floor.pdf`, `figs/expC_bifurcation.pdf`, `figs/expA_dissociation.pdf`.

### Exp C — P2 bifurcation (figs/expC_bifurcation.pdf, results/expC/summary_expC.json)

Decision budget: 100k steps for the cells extended by amendment 3 (all
DeltaNet/DeltaProduct general-stream cells; c_delta_b1, c_diag, c_attn8 both
streams; c_lstm general), 20k otherwise. 20k results retained and reported.
"Pass" = last-position acc ≥ 0.9; cell passes if ≥2/3 seeds pass.

| # | Prediction | Verdict | Numbers (acc@T=256 unless noted; 3 seeds/cell) |
|---|---|---|---|
| C-i (*) | DeltaNet β∈(0,1) fails both streams | **SUPPORTED** | swap 0.011 (max seed 0.017), general 0.008 — chance ≈ 0.008, at 100k. Stronger than predicted: chance even at T=40 (≤0.034), i.e. it never fits the *training* distribution |
| C-ii | DeltaNet β∈(0,2) n_h=1 passes swap | **SUPPORTED** | 3/3 seeds: 1.0 / 1.0 / 0.998 (20k). At T=512 (12.8×): 0.821 mean — degrading |
| C-iii | n_h staircase on general; n_h=4 pass, n_h=1 fail | **SUPPORTED** (100k, 3 seeds; §7a: 3/8 at 8 seeds — below the frozen ratio) | n_h=1: 0.010; n_h=2: 0.011; n_h=4: seeds {0.995, 1.000, 0.013} → 2/3 pass. Monotone non-decreasing ✓. Caveats: (a) the "staircase" is a step — n_h=2 gains nothing; (b) dp4 seed 2 is a convergence failure (chance at T=40 after 100k steps); (c) at 20k only one dp4 seed reached 0.515@256 — budget, not capacity, was binding, exactly as amendment 3 anticipated. dp4 seed 1 generalizes perfectly to T=512 (acc 1.0, 12.8× train length) |
| C-iv | Transformers fail @256 (both streams, both depths) | **SUPPORTED** (amended: 8L control added) | 2L: chance everywhere incl. T=40. 8L (4× params, 100k): swap in-length 1.0 → **0.029 at T=64** (immediate collapse); general: never learns even in-length (best seed 0.413@40, chance beyond). The "high in-length" premise of the prediction is realized only on swap |
| C-v | Diagonal SSM fails @256 | **SUPPORTED** | chance on all cells, both streams, both budgets (≤0.022 at any T) |
| C-vi | LSTM passes both streams @256 | **SUPPORTED** | swap (20k): 3/3 seeds 1.0 at every T incl. 512. General (100k): 3/3 seeds 1.000 at every T incl. 512. At 20k general 0/3 passed (0.594 best @256) — again optimization speed, not capacity |

Engineering notes for C: (1) `summary_expC.json` as produced by the pipeline was
**wrong for C-iii** (reported FAILED): `plot_c.py` loaded 20k and 100k files in
nondeterministic glob order, mixing budgets across seeds. Fixed (100k now
deterministically preferred, per amendment 3) and regenerated; C-iii flips to
SUPPORTED. (2) Controls are width-matched (d_model=128, 2 layers) but not
param-matched: LSTM 1.02M vs DeltaNet 0.25M, diag 0.43M, attn8 1.09M
(declared). This favors predicted-pass LSTM; it does not weaken the fail
verdicts (diag and attn8 fail *with more* parameters than the passing
DeltaProduct).

### Exp B — P1 floor & scissors (figs/expB_floor.pdf, results/expB/summary_expB.json)

All cells 12k steps, dense LM+answer loss (amendment 4), L=512, 3 seeds,
eval N ∈ {2,…,128}, b=8 bits. p_eff fit once on m=256 (frozen): **0.389
bits/scalar**. Error is exact-retrieval error at the answer position.

| # | Prediction | Verdict | Numbers |
|---|---|---|---|
| B-i | Floor rise for every pure-state m | **SUPPORTED** | Monotone rise in N for all m; err > 0.5 well before the pre-registered x=4 (m=64: 0.67 at N=4; m=256: 0.64 at N=16; m=1024: 0.62 at N=64 — all x≈1.3); saturation at N=128: 0.994 / 0.977 / 0.854 (m=1024 still climbing at grid edge; chance = 0.996) |
| B-ii | Load-ratio collapse within ±0.1 at shared x | **PARTIAL** | 50%-crossings at x = 0.82 / 1.00 / 1.03 (m=64/256/1024) — within 1.25×. m=256 vs m=1024 collapse: max deviation **0.024** ✓. But m=64 deviates up to **0.15** from the others (0.09–0.11 after dropping its one non-converged seed), violating the pre-registered ±0.1 band. Strict criterion as frozen: fails; collapse holds cleanly for the two larger m |
| B-iii | Scissors: hybrid m=64 err ≤ 0.05 all N; pure m=64 > 0.5 | **PARTIAL** (near miss; §7b: SUPPORTED at 24k) | Pure m=64 @N=128: **0.994**. Hybrid (same 64-scalar state + 1 global-attn layer): ≤ 0.021 up to N=64, but **0.060 at N=128** (seeds 0.018 / 0.125 / 0.038) — crosses the frozen 0.05 line at the last grid point. The measured scissors is a 0.994-vs-0.060 gap; the pre-registered threshold is missed by 0.01 on the mean (one seed at 0.125) |
| B-iv | Horizon wall: cliff, not slope (amended: mixed-layout SWA) | **PARTIAL** | Far bindings: err 0.994–0.999 at **every** N (= chance; the wall) ✓. Near bindings: 0.069/0.501/0.067 at N=2 per seed, rising with N (mean 0.362 over N≤8 vs frozen ≤0.2; 0.191 excluding the one partially-converged seed). The cliff *exists* (0.93 near-vs-far gap at N=2) but "near-perfect inside window, independent of N" is not realizable as designed: ≳8 bindings (24+ tokens) no longer fit inside w=32. Original far-only-trained SWA: err ≈ 0.995 everywhere incl. the near probe — the horizon wall bites at training time (amendment 6) |
| B-v | No architecture beats the nominal-capacity Fano floor | **SUPPORTED** | No violation at any (arch, N); floor computed with B = 16m bits |
| B-vi | κ ordering at matched weights (amendment 5: mixed-κ-trained m=256) | **PARTIAL — null result** | eval-high vs eval-low on same weights: mean err 0.610 vs 0.609 (per-seed Δ = −0.002 / +0.001 / +0.003). The inequality is not violated, but there is **no detectable κ effect** — the predicted semantic-overlap interference is absent at this scale/design. Per-κ-trained comparison (as originally designed) shows the *opposite* ordering (low-κ-trained worse at small N: optimization confound), reported as failed-as-designed per amendment 5. "Gap shrinks for hybrid" untestable (no mixed-κ hybrid runs) |
| B-vii | Wall bites on effective, not nominal, entropy | **PARTIAL** | c=16: 50%-crossing at N≈27.4 vs effective-entropy prediction 24.9 (nominal predicts 12.4) ✓. c=4: crossing at N≈11.5 ≈ nominal 12.4, not the effective 16.6 ✗. Also the block design admits a retrieval-free shortcut for N ≤ c (only one distinct value present), which contaminates the small-N region — see anomaly (3) below. Exact analytic H per condition was never emitted by the pipeline (deviation) |
| B-viii | Local-window hybrid NOT rescued | **SUPPORTED** | state m=64 + window-32 layer: err 0.995 at N=128 (and ≥ 0.91 at *all* N — it never learns retrieval at all, like the far-trained SWA; a stronger failure than the predicted floor-tracking) |

Convergence-failure seeds vs capacity effects:
- **m=64: 3 of 6 seeds never learned even N=2** (err 0.867 high-κ s2; 0.979,
  0.780 low-κ s0,s2). The B-i/B-ii/B-iii m=64 means above include these seeds
  (no post-hoc exclusion, per reporting commitment 1); numbers excluding them
  are given where they change a verdict's margin.
- **Larger m do NOT share the total-failure mode**: worst N=2 error over all
  m=256/m=1024 seeds is 0.11 (m=256 low-κ s2); mixed-κ m=256 s2 converged to a
  weaker solution (N=2 err 0.269). m=1024: all seeds ≤ 0.098.
- The hybrid and full-attention cells: 0 convergence failures (N=2 err = 0.000
  everywhere).

Anomalies (Exp B):
1. **Full attention is not the upper bound it was cast as**: err rises 0.06 →
   0.91 over N=2→128 (both κ), far *worse* than the hybrid (0.06 at N=128) at
   comparable params (349k vs 289k) and identical budget. A 2-layer full-attn
   model evidently fails to learn composite-key retrieval at high load under
   this budget; the access-complete *witness* in this experiment is the hybrid,
   not full attention. B-i–B-iii verdicts are unaffected (they compare pure
   state vs hybrid), but "full attention (upper-bound reference)" language must
   not survive into the paper.
2. `plot_b.py` had an f-string syntax error that killed the pipeline's plotting
   step — this is why `summary_expB.json`/`expB_floor.pdf` were missing. Fixed
   and regenerated; no training job was missing.
3. **vb16 hybrid found a degenerate shortcut**: with value blocks c=16, error is
   0.000 for N ≤ 16 (only one distinct value in the stream — copying "the"
   value suffices) then jumps to ≈ 1 − 1/(N/16) (0.484 / 0.742 / 0.868 at
   N=32/64/128 ≈ guessing among the distinct values present). The shortcut
   removes the gradient signal for retrieval. This contaminates the vb16 arm of
   B-vii and explains the high "err" of vb16 hybrid checkpoints in Exp D.

### Exp A — dissociation (figs/expA_dissociation.pdf, results/expA/summary_expA.json)

Deviations from protocol (all forced by model context limits / cost, decided
before analysis): L ∈ {896, 1900} instead of 2048 (RWKV-4 pile models were
trained at ctx 1024; 1900 < 2048 leaves tokenizer margin for the largest
models); 100 instances/cell instead of ≥200; the four smallest models were run
at L=896 only. Single evaluation pass per cell (pretrained public weights — no
training seeds; the "3 seeds" convention does not apply to A).

| # | Prediction | Verdict | Numbers |
|---|---|---|---|
| A-i | Cross-family alignment grows with scale (PRH replication) | **SUPPORTED** | Mutual-kNN (k=10, 1024 Pile snippets, max over layer pairs) vs min(params) of pair: Spearman ρ = 0.574 (n=30 pythia–mamba pairs, p=0.0009), 0.868 (n=24 pythia–rwkv, p<1e-4), 0.513 (n=20 mamba–rwkv, p=0.021). Alignment 0.49–0.73 vs chance 0.0098 |
| A-ii | Pythia exceeds Mamba AND RWKV by ≥30pp at N≥16, early depth, matched 1.4–2.8B scale | **PARTIAL** (frozen threshold missed vs Mamba; §7c: confirmed at n=200) | N=16, d=0.1, 2.8B-class: Pythia 0.39 / 0.35 (L896/L1900), Mamba 0.14 / 0.08, RWKV 0.00 / 0.00. Gap vs RWKV: 39pp / 35pp ✓. Gap vs Mamba: **25pp / 27pp — below the frozen 30pp** ✗ (N=32, L1900: 27pp; N=64: 22pp). Stratification direction (Pythia > Mamba > RWKV) holds in *every* matched-scale cell at N≥16, but the margin criterion fails vs Mamba, mainly because Pythia's own accuracy is only 0.35–0.39 at this task difficulty |
| A-iii | Mamba/RWKV recency signature; Pythia flat | **PARTIAL** | Mean acc over N, d0.1→d0.9 at L1900: Mamba +0.09…+0.31 per model (relative 2.2–6.1×; 1.4b: 0.060→0.366) — recency confirmed for Mamba. Pythia-2.8b: +0.042 (1.10×, flat ✓), but smaller Pythias have real slopes (+0.15…+0.23). RWKV: floor effect — acc ≈ 0 at all depths, slope unmeasurable (its only nonzero cells are at d0.9, direction-consistent) |
| A-iv | Alignment does NOT predict capability gap (dissociation quadrant occupied) | **SUPPORTED** | Spearman(alignment, \|gap\|) over 74 cross-family pairs: **ρ = +0.477, p<1e-4 — significantly positive**, not negative (frozen criterion: "not significantly negative"). 21 pairs sit in the high-alignment (>median 0.669) / large-gap (≥0.2) dissociation quadrant. Caveat: the positive sign is partly a scale confound (alignment and gap both grow with scale); the pre-registered directional claim is met regardless |

### Exp D — commensurability miniature (results/expD/d_comm.json)

| # | Prediction | Verdict | Numbers |
|---|---|---|---|
| D-i | Solving hybrids align above random-init floor; failing do not | **FAILED — direction reversed** (§7d: confirmed, also under CKA) | Mutual-kNN (k=10) between state-channel and attention-channel outputs, N=32 probe: solving finals (err ≤ 0.125) **0.283** mean (n=9 ckpts: range 0.212–0.358); failing finals (vb16, err ≥ 0.87) 0.376 (n=3); early failing ckpts 0.527 (n=12); **random init 0.576** (n=3). Training *decreases* cross-channel alignment, and solving models diverge the most. The premise of the schematic Fig. 4 — that random init is a low-alignment floor — is empirically wrong here: two random channels fed the same tokens are highly kNN-aligned (~0.58) because both echo input geometry. P3 as operationalized is contradicted at this scale; per reporting commitment 2, this is a failed CCH prediction |

Exp D protocol notes: (1) the pre-registered "failing" arm was to include
β∈(0,1)-state hybrid variants from C; these were never run (deviation) — the
failing arm here is undertrained checkpoints + the degenerate vb16 finals.
(2) `d_commensurability.py` also swept `b_hybrid_local` checkpoints but loaded
them under the *global*-attention spec (identical state-dict keys, different
runtime mask) — those rows are architecture-mismatched and are excluded from
the numbers above; they are outside the pre-registered D design anyway.
Excluding/including them does not change the verdict.

## Follow-up phase (amendment 7) — results

All four extensions ran 2026-07-10. Analysis
code: `analyze_followup.py` → `results/followup_summary.json`. Frozen first-phase
files are untouched; follow-up outputs live in `results/expC/` (new seeds),
`results/expB/followup/`, `results/expA/followup/`, `results/expD/d_comm_v2.json`.

### 7a — C-iii at 8 seeds: the amended criterion FAILS (3/8 pass)

acc@256 per seed (c_dp4 general, 100k): s0 0.995, s1 1.000, s2 0.013, s3 0.492,
s4 0.474, s5 0.008, s6 0.008, s7 0.995 → **3/8 ≥ 0.9** (frozen ≥2/3 needs 6/8).
The failure modes split three ways: 3 seeds solve and extrapolate (0.50–1.00 at
T=512); **s4 fits in-length perfectly (1.000 @T=40) but collapses at 256 (0.474)**
— the Transformer-like failure shape, new at this cell; s3 is a partial learner
(0.766 @40); s2/s5/s6 never leave chance at T=40 after 100k steps.
Consequence for the paper: C-iii must be stated as a **capacity/existence** result
— n_h=4 is the only architecture in {1,2,4} that ever solves general@256 (3 seeds
at ≈1.0, incl. 12.8× length extrapolation), while n_h≤2 never exceeds 0.022 in any
of 6 seeds; endpoint separation and monotone means (0.010 / 0.011 / 0.498) hold —
with **trainability at this budget explicitly brittle (3/8)**. "Reliably learns"
is not supported. The 3-seed frozen verdict (SUPPORTED) remains the pre-registered
primary per amendment 7a, but an honest paper reports the 8-seed fraction next to it.

### 7b — B-iii at 24k: SUPPORTED at the amended budget

b_hybrid high-κ vb1, 24k steps, 6 seeds: mean err ≤ **0.0003** at every N; worst
seed at N=128 is 0.001 (12k reference: mean 0.060 at N=128, worst seed 0.125).
The 12k miss was undertraining, as suspected. Measured scissors at matched state:
pure m=64 **0.994** vs hybrid **0.000** at N=128. The asymmetric budget cannot be
attacked: the pure-state side sits at chance (0.994 ≈ 0.996) against an
information floor (x ≈ 16 at N=128) that more optimization cannot beat, and the
frozen criterion's failing side (pure) kept its original budget. The 12k PARTIAL
stays the pre-registered primary; supported-at-24k is reported as the amendment.

### 7c — A1 at n=200: the A-ii miss is confirmed; stratification unanimous

Gap vs Mamba (d0.1, exact match): 2.8B class 15.0–**29.5**pp (max at L1900 N=16),
1.4B class 11.0–14.5pp — the frozen ≥30pp is not met in any cell, now with n=200
instances (the near-miss is not an n=100 artifact). Gap vs RWKV: 32.0–40.5pp at
N=16 (≥30 ✓) but 14.0–31.0 at N≥32. Direction Pythia > Mamba > RWKV holds in all
10 matched-scale cells (both classes, both lengths, N∈{16,32,64}) — **A-ii stays
PARTIAL** with tighter uncertainty. A-iii at n=200, L1900: Mamba recency at every
size (mean acc d0.1→d0.9: 1.4b 0.065→0.326; 2.8b 0.046→0.125), Pythia-2.8b flat
(0.441→0.452), smaller Pythias still slope (1.4b 0.250→0.418), RWKV floor (≈0
everywhere; 3b 0.000→0.025) — same PARTIAL, cleaner support for the matched-scale
flagship cell. The four small-model L1900 cells (incl. their N=64 cells) are now
filled per the pre-registered grid.

### 7d — Exp D v2: the D-i reversal is confirmed and strengthened

Re-measurement with the arch inferred from the filename (fixes the v1
b_hybrid_local wrong-mask load), plus the pre-registered β∈(0,1) arm, plus linear
CKA (post-hoc). Mutual-kNN means (N=32 probe): random init **0.526–0.576**
(ceiling, all arch variants) > early ckpts 0.489–0.626 > failing finals
0.388 (vb16) / 0.481 (hybrid_local, correct spec) > **solving finals 0.231–0.284**.
The 24k solving finals (0.231) sit *below* the 12k ones (0.284): longer training →
stronger channel differentiation. CKA is more dramatic: random **0.993** → early
0.71–0.80 → solving finals 0.13–0.33. Both metrics agree: **cross-channel
alignment decreases monotonically with training progress and with task success;
random init is the ceiling, not the floor** — the premise of schematic Fig. 4 is
wrong under both metrics at this scale. Two protocol notes: (1) the pre-registered
"failing arm" b_hybrid_b1 does NOT fail NA (err 0.016; β∈(0,1) breaks S5 state
tracking, not attention-mediated retrieval) — it lands in the solving group and
shows the same low alignment (0.261); the design produced no failing-final hybrid
beyond vb16/hybrid_local. (2) v1's excluded hybrid_local rows are now validly
measured (0.481). **D-i stays FAILED**; the coherent positive finding is channel
*differentiation*, monotone in training and success.

### 8 — Assumption-sep negative control (2026-07-11, run in response to peer review)

Collision arms (`--collide c`): every planted key bound c+1 times to distinct
values, triples shuffled, labeled value a uniformly random one of them —
code distance zero, Bayes floor err = c/(c+1), nll = log2(c+1) bits, N-independent.
b_hybrid, 24k steps, 3 seeds × c ∈ {1,3,7}. Analysis: `analyze_negctrl.py` →
`results/expB/followup/negctrl_summary.json`.

- **Error criterion: SUPPORTED at every c.** Worst-cell |mean − floor| =
  0.023 / 0.025 / 0.005 vs floors 0.500 / 0.750 / 0.875 (frozen band ±0.05).
- **Log-loss band (±0.4 bits): holds at c=1 (+0.25 worst), exceeded at c=3
  (+0.66) and c=7 (+0.89).** The excess is always ABOVE floor (safe direction),
  grows with c and N — a calibration gap (imperfect posterior over the
  candidate set), not a retrieval gap.
- **Red flags: neither fired.** No cell drifts toward chance (0.996) — the
  retrieval machinery still isolates the candidate set without separability.
  Two cells sat 0.023–0.025 BELOW floor, triggering the pre-registered bug
  clause; checked and cleared: fixed deterministic candidate-pickers on the
  same finite eval sets realize deviations spanning ±0.023 (label uniformity
  among candidates verified analytically), so the readings are fixed-set
  sampling fluctuation.
- **Reading for the paper**: the hybrid's scissors advantage is purchased
  exactly where Assumption sep says — in write-time code separability — and
  vanishes bit-for-bit when separability is removed. Integrated as item (ix)
  of the Exp B results in CCH_arxiv.tex §sec:expB.

### 9/9b/10 — The conjunction witness (Exp E) and channel complementarity (D-ii), 2026-07-12

Task: bindings [perm-address → 8-bit value] first, S5 swap-instruction stream drives
the referent, query = value at the COMPOSED final referent. 5 frozen arms × 3 seeds
× 120k steps (amendment 9b records the full pilot history: tying harms; answer_w=8 +
stretched schedule harms; 1-delta-layer hybrids lose tracking to an in-length
attention shortcut; dk16 recurrences never learn tracking). Analysis:
`analyze_expE.py` → `results/expE/summary_expE.json`.

- **WITNESS CRITERION MET**: 7 cells where e_hybrid_d2a passes (≥2/3 seeds ≥0.9)
  and every other arm fails — incl. the predicted (T256, N16): hybrid
  0.966/0.979/0.964. Graded margins at T256 (hybrid − best other): +0.28/+0.42/+0.53
  for N=4/16/64; at (T256,N64) hybrid mean 0.801 (below 0.9 → graded separation per
  9b clause).
- Per-arm signatures, all as predicted (with two honest exceptions): e_state3
  extrapolates perfectly in T, degrades monotonically in N (capacity); e_attn8
  never fits even in-length at 120k (its extrapolation clause therefore
  unfalsifiable here — E-iii holds only in its fail half); e_hybrid_b1_d2a loses
  BOTH axes (sub-prediction "keeps in-length retrieval" WRONG — the composed
  address couples the axes, retrieval inherits the tracking failure); local_d2a
  passes only tiny-N via internal state. 3/15 runs convergence failures (included).
- **D-ii (P3′, corrected script — the original assumed 2-layer hybrids and probed
  the wrong block on 3-layer arms)**: joint necessity — either-channel lesion
  collapses ALL cells (drops ≥0.96, 3/3 seeds); value-side localization exact
  (attn-mixer probe 1.0 vs state 0.003); differential sub-criteria FAILED (axes
  coupled through the computed address; referent linearly decodable from the last
  recurrent mixer in 1/3 seeds only). P3′ = PARTIAL, strongest component holds.
- A-ii n=1000 (amendment 11) also complete: flagship cell 32.0±3.3pp (Mamba) /
  36.5±3.0pp (RWKV) — both above the frozen 30pp; all other cells clearly below.

### 9c — attn8 budget extension verdict (2026-07-12)

e_attn8 s0 at 240k steps (2x every other arm): T40 = 0.27/0.088/0.031 — still at
the 1/N baseline in-length. Per the frozen 9c rule: in-length competence NOT
reached → E-iii remains supported only in its fail half (extrapolation clause
unfalsifiable), now with the stronger statement that the failure is not a budget
artifact. File: results/expE/e_attn8_240k_s0.json (outside the frozen 120k
aggregation). Integrated into CCH_arxiv.tex (main summary + App F Exp E).

### Amended scorecard

Counts are unchanged — **11 SUPPORTED / 7 PARTIAL / 1 FAILED** — but two
predictions swap places under the amendments: B-iii moves up (PARTIAL → supported
at 24k), C-iii moves down (SUPPORTED at 3 seeds → 3/8 at 8 seeds, below the frozen
ratio). Per-prediction status should be read from the sections above, not the counts.

## Deviations & engineering notes
- **Deviations from the frozen protocol not covered by a logged amendment**
  (collected here for the paper's deviations paragraph):
  1. Exp B sliding window w=32, not the pre-registered w=64 (receptive field
     still ≪ binding–query distance; the horizon-wall logic is unchanged).
  2. Exp A: L ∈ {896, 1900} not 2048; 100 instances/cell not ≥200; smallest 4
     models evaluated at L=896 only.
  3. Exp B compressibility axis: exact analytic H_eff per block condition was
     never computed/reported (only the b − log2(c) approximation is used in
     analysis).
  4. Exp D: β∈(0,1)-state hybrid variants (pre-registered as part of the
     failing arm) were never run.
- **Analysis-stage code fixes (post-data, affect no training run)**: f-string
  syntax error in `plot_b.py` (killed the pipeline's Exp B figure/summary step);
  nondeterministic 20k/100k mixing in `plot_c.py`'s loader (produced a wrong
  FAILED verdict for C-iii in the first `summary_expC.json`). Both fixed
  2026-07-10; figures and summaries regenerated from unchanged raw JSONs.
- mamba-ssm + causal-conv1d CUDA kernels compiled from source (torch 2.11/cu130,
  sm_103), so Mamba evals use the fast path; RWKV uses the HF JIT CUDA kernel.
- Chunked WY delta scan: 29× speedup over the sequential reference; exactness
  verified (fwd 1e-17, grad 1e-16, n_h ∈ {1,3,4}).

## Collision-naturalness scoping measurement (exploratory, post-hoc; 2026-07-13)
First surface-form run of the CTA protocol for Assumption 1 naturalness
(`collision_naturalness/measure_collisions.py`, results in `results_collisions.json`).
Windows = pile-10k documents / numpy+scipy .py files (and 8x pooling); near-collision =
char-trigram Jaccard >= 0.5 (numeric IDs: same length, Hamming <= 1).
- Text entities: separable 87.5% (doc, n=8274 windows), 86.5% (8-doc)
- Text numeric IDs: separable 50.7% (doc), 46.8% (8-doc)
- Code identifiers: separable 56.2% (file, n=1096), 54.6% (8-file)
Verdict: Assumption 1 plausible for entity-style keys, materially strained for
morphologically regular keys (IDs, identifiers) — the loads where the collision control
predicts hybrid advantage shrinks toward the ambiguity floor. Surface-form proxy only;
embedding-level version open. NOT pre-registered; reported as scoping, not verdict.
