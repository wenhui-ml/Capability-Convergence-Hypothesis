# CCH Experiments — Pre-Registered Protocol and Decision Criteria

**Date frozen: 2026-07-10, before any full-scale run.**
All support/falsification criteria below were written before the corresponding data
were collected. Smoke tests (tiny configs, used only to verify code runs) are exempt.
Any deviation from this protocol will be reported as a deviation.

Paper: *The Capability Convergence Hypothesis* (CCH_arxiv.tex, this repo).
Predictions under test: **P1** (information floor / scissors gap, Fig. 3 = fig3_fano_floor),
**P2** (state-tracking bifurcation, Fig. 8 = table2_nh_grid), **P3** (channel
commensurability, Fig. 4 = fig4_commensurability), plus the **dissociation experiment**
(representational convergence without capability convergence, new figure).

Common regime (matches the paper's regime R̄): fixed depth, no chain-of-thought,
fixed per-token compute, next-token / classification readout only. All models trained
from scratch use identical training budgets, optimizer, and data distribution within an
experiment; only the sequence-mixing operator differs.

---

## Experiment C — P2: S5 state-tracking bifurcation

**Task.** Word problem over S5. A sequence of group elements g_1 … g_T (tokens);
label at position t is the cumulative product g_1∘…∘g_t (120-way classification at
every position). Two difficulty axes:
- **swap stream**: every token is a single transposition (10 generators);
- **general stream**: every token is an arbitrary S5 element (needs ≤4 transpositions).

**Training**: lengths uniform in [3, 40]; **evaluation**: last-position accuracy at
T ∈ {40, 64, 128, 256, 512}. "Pass" at T := accuracy ≥ 0.9 at that length
(chance = 1/120 ≈ 0.008). "Length-generalizes" := pass at T = 256 (6.4× train length).
3 seeds per cell; a cell passes if ≥2/3 seeds pass (best-of-seeds reported alongside).

**Architectures** (2 layers, matched width/params, identical training):
1. DeltaNet, β∈(0,1) (eigenvalues in (0,1): no reflection) — *the production
   linear-mixer parameterization; the paper's starred prediction*
2. DeltaNet, β∈(0,2) (eigenvalues in [−1,1]: one reflection/token), n_h=1
3. DeltaProduct n_h=2, β∈(0,2)
4. DeltaProduct n_h=4, β∈(0,2)
5. Diagonal SSM (Mamba-style gated diagonal recurrence) — control, predicted fail
6. Transformer (RoPE, full attention) — control, predicted extrapolation failure
7. LSTM — control, predicted pass (nonlinear full-rank recurrence)

**Pre-registered predictions (from paper Fig. 8 / Prop. 4 / App. A.2):**
- C-i (starred *): DeltaNet β∈(0,1) FAILS length generalization on both streams.
- C-ii: DeltaNet β∈(0,2), n_h=1 PASSES swap stream at 256.
- C-iii: staircase in n_h on the general stream: accuracy at 256 non-decreasing in
  n_h ∈ {1,2,4}, with n_h=4 passing and n_h=1 failing.
- C-iv: Transformer fails at 256 on both streams (in-length ≤40 performance may be high).
- C-v: Diagonal SSM fails at 256 on both streams.
- C-vi: LSTM passes at 256 on both streams.

**Falsification:** β∈(0,1) length-generalizing composed S5 (kills C-i, the paper's
sharpest claim); or no n_h staircase (monotonicity violated by >1 grid cell); or
diagonal SSM passing at 256. Any of these → the corresponding paper claim must be
retracted or weakened, and we will say so.

---

## Experiment B — P1: information floor & scissors gap

**Task (executable NA(N, b, B; κ)).** Stream = N key–value bindings (keys drawn
without replacement from a key vocabulary partitioned into semantic clusters; values
uniform over 2^b tokens, b = 8), followed by distractor tokens up to total length
L = 512, then a query key; the model must emit the bound value (next-token readout).
Distractor kernel κ: **high-κ** distractors are unbound keys drawn from the *same*
clusters as the planted keys; **low-κ** from disjoint clusters.
Training: N uniform on a mixed grid up to N_max=128; evaluation per
N ∈ {2,4,8,16,32,64,128}. 3 seeds. Metrics: exact-retrieval error and conditional
log-loss (nats→bits) on the value token.

**Architectures** (matched embedding/MLP; only the mixer differs):
1. Pure state (DeltaNet), total recurrent state m ∈ {64, 256, 1024} scalars (sweep)
2. Sliding-window attention, window w = 64 ≪ binding positions (horizon control)
3. Hybrid: same DeltaNet state (smallest m = 64) + ONE full-attention layer
4. Full attention (upper-bound reference)

**Pre-registered predictions (Theorem 3.3 + Fig. 3):**
- B-i (floor rise): for every pure-state m, exact-retrieval error is (statistically)
  non-decreasing in N and exceeds 50% once Nb ≥ 4·m·p_eff (p_eff defined below),
  saturating toward the b-bit chance floor.
- B-ii (load-ratio collapse): plotting error vs x = Nb/(m·p_eff) with a single
  fitted constant p_eff (effective bits per state scalar, fit once on m=256 and
  frozen for the other m) collapses the three pure-state curves onto one master
  curve within ±0.1 error at each shared x. This is the operational content of
  "error is a function of the load ratio alone".
- B-iii (scissors gap): the hybrid with the *same* m=64 state stays at error ≤ 0.05
  for all N ≤ 128 (its Nb load rides the index channel), while pure m=64 exceeds
  0.5 — the measured scissors of Fig. 3.
- B-iv (horizon wall): sliding-window model is near-perfect for bindings inside w
  and at chance for bindings outside w, independent of N (cliff, not slope).
- B-v (Fano sanity): no architecture beats the Fano floor computed with nominal
  capacity B = 16m bits (bf16) — a hard lower-bound check.
- B-vi (κ ordering): at matched (N, m), high-κ error ≥ low-κ error for pure state
  models; gap shrinks for the hybrid.

**Compressibility axis (answers the "natural bindings are correlated" objection).**
Additional condition: values duplicated across keys in blocks of size c ∈ {1, 4, 16}
(effective entropy per binding ≈ b − log2(c) adjusted for block structure; exact
H computed analytically and reported). Prediction B-vii: pure-state error tracks
**effective** entropy N·H_eff/(m·p_eff), not nominal Nb — i.e. the wall bites only
in the incompressible regime, exactly as Limitation (iii) and the Shannon-wall
independence assumption state.

**Falsification:** a pure-state architecture (no scalable index channel) holding
error ≤ 0.1 at x ≥ 8 with the collapse variable, at any m — this would mean its
effective state is not o(Nb) and P1's channel characterization fails; or the hybrid
tracking the pure-state floor (access-completeness does not rescue retrieval).

---

## Experiment A — dissociation: representational convergence ≠ capability convergence

**Models (all trained on The Pile, overlapping scale, public weights):**
- Pythia (attention): 70m, 160m, 410m, 1b, 1.4b, 2.8b
- Mamba (pure SSM): 130m, 370m, 790m, 1.4b, 2.8b
- RWKV-4-pile (linear RNN): 169m, 430m, 1b5, 3b

**A1 capability axis** — Newton's-apple ICL retrieval: N key–value facts (templated
English, e.g. "The code for X is Y.") planted at controlled depth in Pile-style filler
text, single query at the end, L ≤ 2048 (all models' train context). Scan
N ∈ {4,8,16,32,64} and binding depth ∈ {0.1, 0.5, 0.9 of L}. Metrics: exact-match
accuracy on the value's first token and conditional log-prob. ≥200 instances/cell.

**A2 representation axis** — PRH's own metric: mutual-kNN alignment (k=10) and CKA
between all model pairs, per layer (max over layer pairs reported, as in PRH), on
1024 shared Pile-validation snippets, mean-pooled token features per snippet.

**A3 dissociation scatter** — per cross-family pair: (representational alignment,
capability gap at N=32, depth 0.1). Pre-registered predictions:
- A-i: cross-family alignment grows with scale (PRH replication on this triplet).
- A-ii: at matched scale (~1.4–2.8b), Pythia retrieval accuracy exceeds Mamba and
  RWKV by ≥30 percentage points at N≥16 with early-depth bindings, L=2048.
- A-iii: Mamba/RWKV error worsens with binding depth→0 (recency), Pythia flat
  in-window.
- A-iv: representational alignment does NOT predict retrieval gap: across
  cross-family pairs, Spearman correlation between alignment and |capability gap|
  is not significantly negative (i.e. closest-aligned pairs do not have smallest
  gaps). Equivalently the dissociation quadrant (high alignment, large gap) is
  occupied.
**Falsification:** Mamba/RWKV matching Pythia at N≥16 (no capability stratification
despite architecture) — this would directly contradict CCH's access-structure claim
at the model-family level; or alignment strongly predicting capability (capability
convergence comes free with representational convergence — PRH suffices, CCH adds
nothing).

---

## Experiment D — P3: channel commensurability (reduced scope, honest label)

On trained hybrids from Exp. B (3 seeds × {solved: trained-to-convergence} vs
{failing: undertrained checkpoints and β∈(0,1)-state variants from C}): mutual-kNN
alignment between the state-channel layer output and the attention-layer output at
paired positions, vs a random-init baseline.
- D-i: solving hybrids show alignment above random-init floor; failing hybrids do not
  (qualitative positive association, as the paper's Fig. 4 caption states).
This is a miniature (not the full Zamba/Hymba version); it will be labeled as such.

---

## Amendments (logged before the corresponding data existed)
1. **2026-07-10, Exp C**: the 2-layer Transformer control came out at chance even at
   train length (consistent with fixed-depth TC0 limits — 2 layers cannot compose 40
   group elements). To give the paper's prediction C-iv its intended form ("high
   in-length, collapses under extrapolation"), we add a depth-advantaged control
   `c_attn8` (8 layers, ~4x the parameters of the grid models). Prediction unchanged:
   c_attn8 learns T<=40 but fails at T=256. Both controls are reported. Logged before
   any c_attn8 run started.
2. **2026-07-10, Exp B**: added `b_hybrid_local` (same state + sliding-window-only
   index) before any Exp B run completed. Prediction B-viii: it tracks the pure-state
   floor rise (local index is not access-completeness), unlike `b_hybrid`.

3. **2026-07-10, Exp C budget extension** (logged after 20k-step general-stream runs
   showed position-by-position crawl — pos1 learned instantly, pos2 at ~0.3 by step
   4k — i.e. slow optimization, not saturation): the general-stream cells for all
   DeltaNet/DeltaProduct variants, and ALL cells of the predicted-fail architectures
   entering comparative claims (c_delta_b1 both streams, c_diag both streams), get a
   100k-step budget (5x). Decision thresholds unchanged. Within every comparative
   claim the failing side receives >= the budget of the passing side, so a FAIL
   cannot be attributed to budget. 20k results are also retained and reported.

4. **2026-07-10, Exp B training loss** (logged before any full Exp B run completed):
   with answer-only supervision (1 token/sequence) no architecture — including full
   attention — escaped chance in 2k-step diagnostics; the induction mechanism gets
   too little gradient signal. Training loss becomes LM loss over the whole stream
   PLUS the answer loss (dense supervision, standard for MQAR-family tasks; also the
   natural reading of Theorem 3.3's conditional log-loss). Evaluation metrics are
   unchanged (answer position only). Applied uniformly to every architecture.

5. **2026-07-10, Exp B κ confound** (logged after the per-κ-trained sweep, before
   any mixed-κ run): B-vi as designed confounds representation learning with
   inference-time interference — models were TRAINED separately per κ, and
   low-κ-trained models show elevated error at small N (an optimization effect,
   direction opposite to B-vi). The proper test: train on mixed κ, evaluate high
   vs low on the same weights. Added: b_state256 mixed-κ, 3 seeds. B-vi verdict
   will be based on the mixed-κ models; the per-κ-trained result is reported as
   a (failed-as-designed) observation with this explanation.

6. **2026-07-10, Exp B horizon-cliff control** (logged after per-κ sweep, before
   any mixed-layout run): B-iv as designed trained the sliding-window model ONLY
   on far-binding layouts, where every answer is beyond its receptive field — it
   therefore never learns retrieval at all (error ≈ chance even on the eval-only
   near-binding probe). This total failure is itself evidence of the horizon
   wall's training-time bite and is reported as such, but the *cliff* (perfect
   inside window, chance outside) requires training with mixed layouts. Added:
   b_swa --layout mixed, 3 seeds; B-iv verdict based on those runs.

7. **2026-07-10, follow-up phase** (logged after the first full analysis
   (REPORT.md) and before any follow-up run started; all runs on GPUs 0-3,
   frozen first-phase result files untouched). Four small-cost extensions, each
   with its decision rule fixed now:
   (a) **Exp C, C-iii seeds**: +5 seeds (s3-s7) for c_dp4 general @100k (the
       2/3-pass cell containing one convergence-failure seed). The frozen
       3-seed verdict stands as primary; the 8-seed pass fraction (same 0.9
       threshold, same >=2/3 criterion) is reported alongside.
   (b) **Exp B, B-iii budget**: b_hybrid high-kappa vb1 retrained at 24k steps
       (2x), seeds s0-s5, written to results/expB/followup/ so the frozen 12k
       aggregation and figures are untouched. The 12k verdict stands as the
       pre-registered result; the 24k arm tests whether the 0.060@N=128 miss
       is undertraining (supported-at-24k iff mean err <= 0.05 at all N).
   (c) **Exp A, A1 instances**: n_per_cell 100 -> 200 (the pre-registered spec
       was >=200) for all 15 models x L in {896, 1900}; this also fills the
       four missing small-model L1900 cells (adding their N=64 cells).
       A-ii/A-iii re-evaluated at n=200 with unchanged thresholds; n=100
       results retained. Outputs under results/expA/followup/.
   (d) **Exp D protocol completion**: add the pre-registered-but-never-run
       failing arm (b_hybrid_b1: identical hybrid, state-channel beta in
       (0,1)), 3 seeds, 12k; re-measure ALL hybrid checkpoints with the arch
       inferred from the filename (fixes the v1 b_hybrid_local wrong-mask
       load); add linear CKA as a post-hoc secondary metric. d_comm.json (v1)
       stays frozen; v2 output is results/expD/d_comm_v2.json. D-i's FAILED
       verdict on the v1 data stands; v2 is reported as protocol completion,
       whatever it shows.

8. **2026-07-11, Assumption-sep negative control** (logged before any run of this
   arm; motivated by peer review asking whether the hybrid's advantage is
   assumption-laden — this arm tests the assumption's conditional structure
   directly). Task: NA with `collide = c` — every planted key is bound c+1 times
   to DISTINCT values, the c+1 triples shuffled through the binding block, and the
   labeled "true" value is a uniformly random one of them with NO distinguishing
   feature in the stream (code distance between target and colliding distractors
   is exactly zero: the zero-distance limit of Assumption sep's kappa
   distractors). Information-theoretic Bayes floor, architecture-independent:
   err = c/(c+1), value log-loss = log2(c+1) bits, independent of N.
   Arm: b_hybrid, high kappa, 24k steps (matching amendment 7b's clean budget),
   3 seeds x c in {1, 3, 7}; N grid restricted to fit L=512
   (c=1: N<=64, c=3: N<=32, c=7: N<=16). Frozen decision rules:
   - CONTROL SUPPORTED iff for every c, mean err (3 seeds) is within +-0.05 of
     c/(c+1) at every evaluated N, and mean log-loss within +-0.4 bits of
     log2(c+1): the hybrid's retrieval machinery still works (it reaches the
     floor, far from the 0.996 chance level) but separability is what its
     scissors advantage was purchasing (it cannot beat the floor) — the
     conditional structure of Assumption sep, measured.
   - Mean err BELOW floor - 0.02 at any N indicates a task-generation bug (the
     floor is information-theoretic); halt and debug, report the incident.
   - Mean err >= floor + 0.15 at any N (drifting toward chance) falsifies the
     graceful-inheritance framing (collisions would be destroying retrieval
     itself, not just the contraction); reported as such if it occurs.
   Outputs to results/expB/followup/ (collide-tagged filenames); the frozen
   first-phase and amendment-7 aggregations are untouched.

9. **2026-07-11, Experiment E: the composite witness** (task generator and
   supervision design frozen after smoke pilots — pilots exempt per protocol;
   decision criteria below frozen before any counted run). Task NA-composed-
   with-S5: bindings [perm-address -> 8-bit value] written first, a swap-
   instruction stream drives the referent r_t through S5, the query asks for
   the value bound to the COMPOSED final referent r_T (always planted). One
   prediction requires state tracking AND exact retrieval — the direct witness
   of Prop. nonpreserve's strict inclusion. Supervision, uniform across every
   arm: LM over the stream + dense "running answer" labels at instruction
   positions (value bound to current referent when planted, else its address
   token — the composite analogue of Exp C's per-position labels) + weighted
   answer loss. Train N in {4..64}, T in [3,40]; eval T in {40,64,128,256} x
   N in {4,16,64}; pass = acc >= 0.9, cell passes if >= 2/3 seeds pass; 3
   seeds per arm. ARMS AND PREDICTIONS (architecture configs frozen from the
   pilot round; state-channel size chosen so the tracking-capable state is
   still o(Nb) at the top load):
   - E-i e_hybrid (beta(0,2) tracking state + 1 global attn): PASSES the
     witness cell (T=256, N=64) and all easier cells.
   - E-ii e_state (pure recurrence, same total state budget): passes small-N
     long-T cells (tracking intact) but FAILS (T=256, N=64) with acc <= 0.5
     (capacity, Shannon wall) — the load-axis-only failure signature.
   - E-iii e_attn8 (8-layer Transformer, 4x params): passes in-length (T=40)
     cells at small N but FAILS every T=256 cell (extrapolation, circuit wall).
   - E-iv e_hybrid_b1 (beta(0,1) state + attn): FAILS every T=256 cell
     (tracking arm broken) while passing (T=40, N<=64) (retrieval intact).
   - E-v e_hybrid_local (state + window-32): FAILS all cells with N>=16
     (bindings sit beyond the window at every evaluated T).
   - WITNESS CRITERION (the measured strict inclusion): there exists a cell —
     predicted (T=256, N=64) — where e_hybrid passes and EVERY other arm
     fails. If e_hybrid also fails it, the learnability of the witness at this
     scale is refuted as designed and will be reported as such (the existence
     claim of App. A is by construction and unaffected, but the paper's
     learnability bridge loses its strongest plank).
   - Optimization-brittleness clause (C-iii lesson): verdicts use >= 2/3 of 3
     seeds; per-seed values reported; budget identical across arms.

9b. **2026-07-12, Experiment E: final arm configs frozen from the pilot round**
   (per amendment 9's provision; logged before any counted run). Pilot history,
   reported as part of the record: weight tying harms (breaks tracking);
   answer_w=8 with a stretched schedule harms; a 1-delta-layer hybrid learns
   in-length tracking through the ATTENTION layer (C-iv's shortcut inside the
   hybrid) and fails extrapolation; a 3-layer pure recurrence (dk32) learns the
   full conjunction within its capacity and extrapolates perfectly; dk16
   recurrences fail to learn tracking at all (excluded). FROZEN ARMS (all 120k
   steps, answer_w=4, untied, 3 seeds): e_hybrid_d2a (delta32 x2 + global attn),
   e_state3 (delta32 x3, pure), e_attn8 (8-layer attention control),
   e_hybrid_b1_d2a (beta(0,1) recurrent front + attn), e_hybrid_local_d2a
   (recurrent front + window-32). Criteria unchanged from amendment 9. Witness
   cell prediction: (T=256, N=64) — e_hybrid_d2a passes (pilot s0 at 100k:
   0.855, budget extended to 120k), e_state3 fails on the load axis (pilot:
   0.704), all controls fail on their predicted axes. If e_hybrid_d2a lands
   below 0.9 at (T256,N64) but above every other arm, the witness is reported
   as a graded separation rather than a pass/fail witness, per the honest-
   reporting commitments.

9c. **2026-07-12, attn8 budget extension** (logged before the run; motivated by
   peer review noting E-iii's extrapolation clause is unfalsifiable while the
   8-layer control fails even in-length at 120k). One run: e_attn8, seed 0,
   240k steps (2x every other arm — extending only the predicted-fail side, per
   the amendment-3 principle that a FAIL must never be attributable to budget).
   Frozen decision rule: if it reaches in-length competence (T40_N4 >= 0.9),
   E-iii's extrapolation clause becomes testable and is judged on the T256
   cells (predicted: collapse); if it still fails in-length, E-iii remains
   supported only in its fail half and is reported as such. Output:
   results/expE/e_attn8_240k_s0.json (kept out of the frozen 120k aggregation).

10. **2026-07-11, Experiment D-ii: channel complementarity, re-operationalized**
   (the re-operationalization Prediction pred:comm's failure demanded; frozen
   before any D-ii measurement; runs on Exp E's trained e_hybrid checkpoints).
   Two probes at the query/instruction positions of the composite task, using
   the per-channel MIXER outputs (residual-stream deltas, isolating each
   channel's contribution):
   (a) Decodability double dissociation: linear probes from state-mixer vs
       attn-mixer outputs to (i) the current referent r_t (120-way) and
       (ii) the answer value (256-way). PREDICTION: state-mixer decodes r_t
       better than attn-mixer by >= 20pp, attn-mixer decodes the value better
       than state-mixer by >= 20pp (interaction pattern), with each channel's
       primary decode >= 0.5.
   (b) Channel lesion: skip (identity-bypass) one mixer at eval. PREDICTION:
       skipping the state mixer collapses long-T cells (acc drop >= 50pp at
       T=256, N=16) more than it collapses (T=40, N=64); skipping the attn
       mixer collapses high-N cells (drop >= 50pp at T=40, N=64) more than
       (T=256, N=16) — a double dissociation in failure geometry.
   Interpretation commitment: if BOTH hold, pred:comm is re-established as
   channel complementarity (division of labour), replacing the failed
   similarity operationalization; if either fails, that too is reported.

11. **2026-07-11, A-ii statistical resolution**: the six deciding cells
   (pythia-2.8b / mamba-2.8b-hf / rwkv-4-3b-pile and pythia-1.4b /
   mamba-1.4b-hf / rwkv-4-1b5-pile, L=1900) re-evaluated at n_per_cell = 1000
   (5x amendment 7c). The frozen 30pp point criterion is unchanged; we commit
   to reporting the two-proportion 95% CIs and to letting the n=1000 point
   estimate settle the previously straddling verdict in whichever direction it
   falls. Outputs to results/expA/followup/ (n1000-tagged).

## Reporting commitments
1. All cells run to completion are reported; no post-hoc exclusion of seeds/cells.
2. Any prediction that fails is reported as a failed prediction of CCH.
3. Fitted constants: only p_eff (Exp. B), fit once, frozen, reported.
4. Code, seeds, and raw result JSONs are kept under experiments/ for reproduction.
