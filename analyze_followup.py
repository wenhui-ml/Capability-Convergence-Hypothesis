"""Unified analysis of the follow-up phase (PREREGISTRATION.md amendment 7).

Emits results/followup_summary.json with the amended verdicts:
(a) C-iii at 8 seeds, (b) B-iii at the 24k budget, (c) A-ii/A-iii at n=200,
(d) Exp D v2 groups (correct arch specs, kNN + CKA, b_hybrid_b1 arm).
Frozen first-phase verdicts are not recomputed here.
"""
import glob, json, os
import numpy as np

RD = os.path.join(os.path.dirname(__file__), 'results')
NG = [2, 4, 8, 16, 32, 64, 128]
out = {}

# ---- (a) C-iii: c_dp4 general @100k, now 8 seeds ----
seeds = {}
for f in sorted(glob.glob(f'{RD}/expC/c_dp4_general_100k_s*.json')):
    d = json.load(open(f))
    seeds[d['seed']] = d['final_acc']
acc = {T: {s: seeds[s][T] for s in sorted(seeds)} for T in ['40', '256', '512']}
n_pass = sum(a >= 0.9 for a in acc['256'].values())
out['C_iii_8seed'] = dict(
    acc_at_40=acc['40'], acc_at_256=acc['256'], acc_at_512=acc['512'],
    n_pass_at_256=n_pass, n_seeds=len(seeds),
    amended_pass=bool(n_pass >= np.ceil(2 / 3 * len(seeds))))

stair = {}
for arch in ['c_delta_b2', 'c_dp2', 'c_dp4']:
    per = {}
    for f in sorted(glob.glob(f'{RD}/expC/{arch}_general_*.json'),
                    key=lambda p: ('_100k_' in os.path.basename(p), p)):
        d = json.load(open(f))
        per[d['seed']] = d['final_acc']['256']
    stair[arch] = dict(per_seed=per, mean=round(float(np.mean(list(per.values()))), 3))
out['C_staircase_general_256'] = stair

# ---- (b) B-iii: hybrid high-kappa vb1 at 24k, 6 seeds ----
r24 = [json.load(open(f)) for f in
       sorted(glob.glob(f'{RD}/expB/followup/b_hybrid_high_vb1_24k_s*.json'))]
a24 = np.array([[r['final'][str(N)]['err'] for N in NG] for r in r24])
r12 = [json.load(open(f)) for f in
       sorted(glob.glob(f'{RD}/expB/b_hybrid_high_vb1_s*.json'))]
a12 = np.array([[r['final'][str(N)]['err'] for N in NG] for r in r12])
out['B_iii_24k'] = dict(
    seeds=[r['seed'] for r in r24],
    mean_err_by_N={str(N): round(float(m), 4) for N, m in zip(NG, a24.mean(0))},
    err_N128_per_seed={r['seed']: r['final']['128']['err'] for r in r24},
    max_mean_err=round(float(a24.mean(0).max()), 4),
    amended_pass=bool(a24.mean(0).max() <= 0.05),
    ref_12k_err_N128_per_seed={r['seed']: r['final']['128']['err'] for r in r12},
    ref_12k_max_mean_err=round(float(a12.mean(0).max()), 4))

# ---- (c) A1 at n=200 ----
cells = {}
for f in glob.glob(f'{RD}/expA/followup/a1_*_n200.json'):
    d = json.load(open(f))
    cells[(d['model'].split('/')[-1], d['length'])] = d['cells']
classes = {'2.8b': ('pythia-2.8b', 'mamba-2.8b-hf', 'rwkv-4-3b-pile'),
           '1.4b': ('pythia-1.4b', 'mamba-1.4b-hf', 'rwkv-4-1b5-pile')}
aii = {}
for cls, (p, mm, rw) in classes.items():
    for L in [896, 1900]:
        for N in [16, 32, 64]:
            key = f'N{N}_d0.1'
            if any(key not in cells.get((t, L), {}) for t in (p, mm, rw)):
                continue
            pa = cells[(p, L)][key]['acc']
            ma = cells[(mm, L)][key]['acc']
            ra = cells[(rw, L)][key]['acc']
            aii[f'{cls}_L{L}_N{N}'] = dict(
                pythia=pa, mamba=ma, rwkv=ra,
                gap_mamba_pp=round((pa - ma) * 100, 1),
                gap_rwkv_pp=round((pa - ra) * 100, 1))
out['A_ii_n200_d01'] = aii
out['A_ii_n200_pass_30pp'] = bool(aii) and all(
    v['gap_mamba_pp'] >= 30 and v['gap_rwkv_pp'] >= 30 for v in aii.values())

aiii = {}
for (tag, L), cc in sorted(cells.items()):
    if L != 1900:
        continue
    byd = {}
    for d0 in ['0.1', '0.5', '0.9']:
        vals = [v['acc'] for k, v in cc.items() if k.endswith(f'_d{d0}')]
        byd[d0] = round(float(np.mean(vals)), 3)
    aiii[tag] = byd
out['A_iii_meanacc_by_depth_L1900'] = aiii

# ---- (d) Exp D v2 ----
rows = json.load(open(f'{RD}/expD/d_comm_v2.json'))['rows']
groups = {}
for r in rows:
    if r['stage'] == 'final':
        b = ('final_solving' if r['err'] <= 0.125 else
             'final_failing' if r['err'] >= 0.87 else 'final_mid')
    else:
        b = r['stage']
    groups.setdefault((r['arch'], r['budget'] or '-', b), []).append(r)
out['D_v2_groups'] = {
    f'{a}|{bud}|{st}': dict(
        n=len(g),
        knn=round(float(np.mean([x['align_knn'] for x in g])), 4),
        knn_range=[round(min(x['align_knn'] for x in g), 3),
                   round(max(x['align_knn'] for x in g), 3)],
        cka=round(float(np.mean([x['align_cka'] for x in g])), 4),
        err=round(float(np.mean([x['err'] for x in g])), 3))
    for (a, bud, st), g in sorted(groups.items())}

json.dump(out, open(f'{RD}/followup_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
