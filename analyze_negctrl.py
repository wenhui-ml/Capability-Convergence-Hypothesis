"""Amendment 8 analysis: Assumption-sep negative control (collision arms).

Frozen decision rules (PREREGISTRATION.md amendment 8):
  SUPPORTED iff for every c: |mean err - c/(c+1)| <= 0.05 at every N
             and |mean nll - log2(c+1)| <= 0.4 bits at every N.
  err < floor - 0.02 anywhere -> task-generation bug (halt, debug).
  err >= floor + 0.15 anywhere -> graceful-inheritance framing falsified.
"""
import glob, json
import numpy as np

out = {}
for c in [1, 3, 7]:
    fs = sorted(glob.glob(f'results/expB/followup/b_hybrid_high_cx{c}_24k_s*.json'))
    runs = [json.load(open(f)) for f in fs]
    ng = sorted(int(k) for k in runs[0]['final'])
    err = np.array([[r['final'][str(n)]['err'] for n in ng] for r in runs])
    nll = np.array([[r['final'][str(n)]['nll_bits'] for n in ng] for r in runs])
    fe, fn = c / (c + 1), float(np.log2(c + 1))
    me, mn = err.mean(0), nll.mean(0)
    out[c] = dict(
        n_seeds=len(runs), N_grid=ng, floor_err=round(fe, 4), floor_nll=round(fn, 4),
        mean_err_by_N={n: round(float(e), 4) for n, e in zip(ng, me)},
        mean_nll_by_N={n: round(float(x), 4) for n, x in zip(ng, mn)},
        max_abs_dev_err=round(float(np.abs(me - fe).max()), 4),
        max_abs_dev_nll=round(float(np.abs(mn - fn).max()), 4),
        below_floor_flag=bool((me < fe - 0.02).any()),
        toward_chance_flag=bool((me >= fe + 0.15).any()),
        pass_err=bool(np.abs(me - fe).max() <= 0.05),
        pass_nll=bool(np.abs(mn - fn).max() <= 0.4))
out['CONTROL_SUPPORTED'] = all(out[c]['pass_err'] and out[c]['pass_nll']
                               for c in [1, 3, 7])
json.dump(out, open('results/expB/followup/negctrl_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
