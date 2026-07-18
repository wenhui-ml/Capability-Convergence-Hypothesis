"""Exp E verdicts per PREREGISTRATION.md amendment 9.

Pass = acc >= 0.9; a cell passes if >= 2/3 seeds pass. Witness criterion:
a cell where e_hybrid passes and every other arm fails (predicted T256_N64).
"""
import glob, json
import numpy as np

PASS = 0.9
ARMS = ['e_hybrid_d2a', 'e_state3', 'e_attn8', 'e_hybrid_b1_d2a', 'e_hybrid_local_d2a']
CELLS = [f"T{t}_N{n}" for t in [40, 64, 128, 256] for n in [4, 16, 64]]

data = {}
for arm in ARMS:
    runs = [json.load(open(f)) for f in sorted(glob.glob(f'results/expE/{arm}_s*.json'))]
    data[arm] = runs

out = {'per_arm': {}}
for arm, runs in data.items():
    cells = {}
    for c in CELLS:
        vals = [r['final_acc'][c] for r in runs]
        cells[c] = dict(per_seed=[round(v, 3) for v in vals],
                        mean=round(float(np.mean(vals)), 3),
                        cellpass=bool(sum(v >= PASS for v in vals)
                                      >= np.ceil(2 * len(vals) / 3)))
    out['per_arm'][arm] = cells

def cp(arm, cell):
    return out['per_arm'][arm][cell]['cellpass']

v = {}
v['E-i hybrid passes witness cell T256_N64'] = cp('e_hybrid_d2a', 'T256_N64')
v['E-i hybrid passes ALL cells'] = all(cp('e_hybrid_d2a', c) for c in CELLS)
v['E-ii state fails T256_N64 (mean below 0.9 pass line)'] = bool(
    out['per_arm']['e_state3']['T256_N64']['mean'] < PASS)
v['E-ii state passes small-N long-T (T256_N4)'] = cp('e_state3', 'T256_N4')
v['E-iii attn8 passes in-length small-N (T40_N4)'] = cp('e_attn8', 'T40_N4')
v['E-iii attn8 fails every T256 cell'] = all(
    not cp('e_attn8', f'T256_N{n}') for n in [4, 16, 64])
v['E-iv b1 fails every T256 cell'] = all(
    not cp('e_hybrid_b1_d2a', f'T256_N{n}') for n in [4, 16, 64])
v['E-iv b1 passes T40_N64 (retrieval intact)'] = cp('e_hybrid_b1_d2a', 'T40_N64')
v['E-v local fails all N>=16 cells'] = all(
    not cp('e_hybrid_local_d2a', f'T{t}_N{n}') for t in [40, 64, 128, 256]
    for n in [16, 64])
witness = [c for c in CELLS if cp('e_hybrid_d2a', c)
           and all(not cp(a, c) for a in ARMS if a != 'e_hybrid_d2a')]
# graded-separation fallback (amendment 9b): per-cell margins over best non-hybrid
grade = {}
for c in CELLS:
    hyb = out['per_arm']['e_hybrid_d2a'][c]['mean']
    rest = max(out['per_arm'][a][c]['mean'] for a in ARMS if a != 'e_hybrid_d2a')
    grade[c] = round(hyb - rest, 3)
out['graded_margin_hybrid_minus_best_other'] = grade
v['WITNESS cells (hybrid passes, every other arm fails)'] = witness
v['WITNESS criterion met'] = bool(witness)
out['verdicts'] = v
json.dump(out, open('results/expE/summary_expE.json', 'w'), indent=1)
print(json.dumps(v, indent=1))
for arm in ARMS:
    print(arm, {c: out['per_arm'][arm][c]['mean'] for c in
                ['T40_N4', 'T40_N64', 'T256_N4', 'T256_N16', 'T256_N64']})
print('graded margins (hybrid - best other):',
      {c: grade[c] for c in ['T256_N4', 'T256_N16', 'T256_N64']})
