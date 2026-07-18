"""Exp C figures + verdict table: measured version of paper Fig. 8 (table2_nh_grid)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RD = os.path.join(os.path.dirname(__file__), '..', 'results', 'expC')
FD = os.path.join(os.path.dirname(__file__), '..', 'figs')
LENGTHS = [40, 64, 128, 256, 512]
ARCHS = ['c_delta_b1', 'c_delta_b2', 'c_dp2', 'c_dp4', 'c_diag', 'c_attn',
         'c_attn8', 'c_lstm']
LABELS = {'c_delta_b1': 'DeltaNet $\\beta\\in(0,1)$', 'c_delta_b2': 'DeltaNet $\\beta\\in(0,2)$',
          'c_dp2': 'DeltaProduct $n_h{=}2$', 'c_dp4': 'DeltaProduct $n_h{=}4$',
          'c_diag': 'diagonal SSM', 'c_attn': 'Transformer 2L',
          'c_attn8': 'Transformer 8L', 'c_lstm': 'LSTM'}
PASS = 0.9


def load():
    # 20k files load first, then _100k_ files overwrite per-seed: amendment 3
    # extends the budget with decision thresholds unchanged, so verdicts (and the
    # main figure) use the highest-budget run of each (arch, stream, seed).
    data = {}
    files = sorted(glob.glob(os.path.join(RD, 'c_*.json')),
                   key=lambda p: ('_100k_' in os.path.basename(p), p))
    for f in files:
        d = json.load(open(f))
        data.setdefault((d['arch'], d['stream']), {})[d['seed']] = d['final_acc']
    return data


def main():
    data = load()
    os.makedirs(FD, exist_ok=True)
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.62, wspace=0.28)

    # top: accuracy-vs-length curves
    for ci, stream in enumerate(['swap', 'general']):
        ax = fig.add_subplot(gs[0, 2 * ci:2 * ci + 2])
        for arch in ARCHS:
            seeds = data.get((arch, stream), {})
            if not seeds:
                continue
            accs = np.array([[seeds[s][str(T)] for T in LENGTHS]
                             for s in sorted(seeds)])
            mean = accs.mean(0)
            ax.plot(LENGTHS, mean, 'o-', label=LABELS[arch], lw=1.8, ms=4)
            ax.fill_between(LENGTHS, accs.min(0), accs.max(0), alpha=0.15)
        ax.axvline(40, color='gray', ls=':', lw=1)
        ax.text(41, 1.02, 'train limit', fontsize=8, color='gray')
        ax.axhline(1 / 120, color='k', ls='--', lw=0.8)
        ax.text(300, 0.02, 'chance', fontsize=8)
        ax.set_xscale('log')
        ax.set_xticks(LENGTHS)
        ax.set_xticklabels(LENGTHS)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel('evaluation length $T$')
        ax.set_ylabel('last-position accuracy')
        ax.set_title(f"{'(a)' if ci==0 else '(b)'} {stream} stream "
                     f"({'1 transposition/token' if stream=='swap' else 'arbitrary $S_5$ element/token'})")
        if ci == 0:
            top_handles, top_labels = ax.get_legend_handles_labels()

    # bottom: measured pass/fail grid (mean acc annotated)
    for ci, stream in enumerate(['swap', 'general']):
        ax = fig.add_subplot(gs[1, 2 * ci:2 * ci + 2])
        M = np.full((len(ARCHS), len(LENGTHS)), np.nan)
        for ri, arch in enumerate(ARCHS):
            seeds = data.get((arch, stream), {})
            for li, T in enumerate(LENGTHS):
                if seeds:
                    M[ri, li] = np.mean([seeds[s][str(T)] for s in seeds])
        im = ax.imshow(M, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        for ri in range(len(ARCHS)):
            for li in range(len(LENGTHS)):
                if not np.isnan(M[ri, li]):
                    ax.text(li, ri, f"{M[ri,li]:.2f}", ha='center', va='center',
                            fontsize=8,
                            color='black' if 0.25 < M[ri, li] < 0.85 else 'white')
        ax.set_xticks(range(len(LENGTHS)))
        ax.set_xticklabels(LENGTHS)
        ax.set_yticks(range(len(ARCHS)))
        ax.set_yticklabels([LABELS[a] for a in ARCHS] if ci == 0 else [],
                           fontsize=8)
        ax.set_xlabel('evaluation length $T$')
        ax.set_title(f"{'(c)' if ci==0 else '(d)'} {stream}: mean accuracy over seeds")
    # legend in the band BETWEEN the curve row (a,b) and the heatmap row (c,d): it
    # keys the top-row colors and sits with the curves it explains (the heatmaps below
    # already list the same arms on their y-axis, so a bottom legend would be redundant)
    fig.legend(top_handles, top_labels, loc='center', bbox_to_anchor=(0.5, 0.505),
               ncol=8, frameon=False, fontsize=7.5, columnspacing=1.1, handlelength=1.5,
               handletextpad=0.4)
    fig.suptitle('Exp C (P2): state-tracking bifurcation on $S_5$ — measured', y=0.99)
    fig.savefig(os.path.join(FD, 'expC_bifurcation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FD, 'expC_bifurcation.png'), dpi=150, bbox_inches='tight')

    # verdicts vs pre-registered predictions
    def cellpass(arch, stream, T=256):
        # >=2/3 of seeds, generalized past 3 seeds (amendment 7a adds seeds)
        seeds = data.get((arch, stream), {})
        if len(seeds) < 2:
            return None
        # plain bool: verdict lines compare with `is False`, numpy bools break that
        return bool(sum(seeds[s][str(T)] >= PASS for s in seeds)
                    >= np.ceil(2 * len(seeds) / 3))

    v = {}
    v['C-i (starred): delta b(0,1) FAILS both streams @256'] = (
        cellpass('c_delta_b1', 'swap') is False and cellpass('c_delta_b1', 'general') is False)
    v['C-ii: delta b(0,2) nh=1 PASSES swap @256'] = cellpass('c_delta_b2', 'swap')
    g256 = {a: np.mean([data[(a, 'general')][s]['256']
                        for s in data.get((a, 'general'), {})])
            for a in ['c_delta_b2', 'c_dp2', 'c_dp4'] if (a, 'general') in data}
    v['C-iii: nh staircase on general (acc@256 nondecr, nh4 pass, nh1 fail)'] = (
        len(g256) == 3 and g256['c_delta_b2'] <= g256['c_dp2'] + 0.05
        and g256['c_dp2'] <= g256['c_dp4'] + 0.05
        and cellpass('c_dp4', 'general') and cellpass('c_delta_b2', 'general') is False)
    v['C-iv: transformers fail @256 (both streams, both depths)'] = all(
        cellpass(a, s) is False for a in ['c_attn', 'c_attn8'] for s in ['swap', 'general']
        if cellpass(a, s) is not None)
    v['C-v: diagonal SSM fails @256'] = (cellpass('c_diag', 'swap') is False
                                         and cellpass('c_diag', 'general') is False)
    v['C-vi: LSTM passes @256 both streams'] = (cellpass('c_lstm', 'swap')
                                                and cellpass('c_lstm', 'general'))
    json.dump(dict(verdicts={k: bool(x) if x is not None else None for k, x in v.items()},
                   acc_at_256_general=g256,
                   n_seeds={f"{a}/{s}": len(d) for (a, s), d in sorted(data.items())}),
              open(os.path.join(RD, 'summary_expC.json'), 'w'), indent=1)
    for k, x in v.items():
        print(('SUPPORTED' if x else 'FAILED/INCOMPLETE'), '--', k)


if __name__ == '__main__':
    main()
