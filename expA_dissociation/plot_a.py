"""Exp A figures: representational convergence WITHOUT capability convergence.

(a) retrieval accuracy vs params per family  (b) depth/recency signature
(c) PRH replication: cross-family alignment vs scale  (d) dissociation scatter.
"""
import glob, itertools, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RD = os.path.join(os.path.dirname(__file__), '..', 'results', 'expA')
FD = os.path.join(os.path.dirname(__file__), '..', 'figs')

PARAMS = {  # embedding-included sizes, millions
    'pythia-70m': 70, 'pythia-160m': 162, 'pythia-410m': 405, 'pythia-1b': 1011,
    'pythia-1.4b': 1414, 'pythia-2.8b': 2775,
    'mamba-130m-hf': 129, 'mamba-370m-hf': 372, 'mamba-790m-hf': 793,
    'mamba-1.4b-hf': 1372, 'mamba-2.8b-hf': 2768,
    'rwkv-4-169m-pile': 169, 'rwkv-4-430m-pile': 430,
    'rwkv-4-1b5-pile': 1515, 'rwkv-4-3b-pile': 2985}
FAM = {'pythia': ('Pythia (attention)', '#2c3e50', 'o'),
       'mamba': ('Mamba (pure SSM)', '#c0392b', 's'),
       'rwkv': ('RWKV-4 (linear RNN)', '#e67e22', '^')}


def family(name):
    return ('pythia' if 'pythia' in name else 'mamba' if 'mamba' in name else 'rwkv')


def load_a1(length):
    out = {}
    for f in glob.glob(os.path.join(RD, f'a1_*_L{length}.json')):
        d = json.load(open(f))
        out[d['model'].split('/')[-1]] = d['cells']
    return out


def below(ax, ncol):
    """Legend below the panel so it never overlaps the curves (unified style)."""
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=ncol,
              frameon=False, columnspacing=1.3, handlelength=1.6, handletextpad=0.4)


def main():
    os.makedirs(FD, exist_ok=True)
    a1 = load_a1(896)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    plt.subplots_adjust(wspace=0.3, bottom=0.22)

    # (a) capability vs scale (N=16, early depth)
    ax = axes[0]
    for fam, (lab, col, mk) in FAM.items():
        pts = sorted((PARAMS[m], c['N16_d0.1']['acc']) for m, c in a1.items()
                     if family(m) == fam and 'N16_d0.1' in c)
        if pts:
            ax.plot(*zip(*pts), mk + '-', color=col, label=lab, lw=1.8, ms=6)
    ax.set_xscale('log')
    ax.set_xlabel('parameters (M)')
    ax.set_ylabel('retrieval accuracy')
    ax.set_title('(a) capability stratifies by access structure\n'
                 '($N{=}16$ facts, depth 0.1, $L{=}896$, same Pile data+tokenizer)')
    ax.set_ylim(-0.03, 1.0)

    # (b) recency signature: acc vs depth for largest of each family
    ax = axes[1]
    largest = {'pythia': 'pythia-2.8b', 'mamba': 'mamba-2.8b-hf',
               'rwkv': 'rwkv-4-3b-pile'}
    for fam, (lab, col, mk) in FAM.items():
        m = largest[fam]
        if m in a1:
            accs = [a1[m][f'N16_d{d}']['acc'] for d in [0.1, 0.5, 0.9]]
            ax.plot([0.1, 0.5, 0.9], accs, mk + '-', color=col, label=lab, lw=1.8, ms=6)
    ax.set_xlabel('fact depth (0 = context start, 1 = at query)')
    ax.set_ylabel('retrieval accuracy')
    ax.set_title('(b) recency signature at ~3B\n(slope = compressive state, flat = index)')

    # (c)+(d) need alignment
    alnf = os.path.join(RD, 'a2_alignment.json')
    if os.path.exists(alnf):
        aln = json.load(open(alnf))
        pairs = aln['pairs']
        chance = aln['chance_knn']

        # (c) PRH replication: cross-family alignment vs min scale
        ax = axes[2]
        for fa, fb, col in [('pythia', 'mamba', '#8e44ad'),
                            ('pythia', 'rwkv', '#16a085'),
                            ('mamba', 'rwkv', '#7f8c8d')]:
            pts = []
            for key, v in pairs.items():
                a, b = key.split('__')
                if {family(a), family(b)} == {fa, fb}:
                    pts.append((min(PARAMS[a], PARAMS[b]), v['mutual_knn']))
            if pts:
                pts.sort()
                x, y = zip(*pts)
                ax.scatter(x, y, s=18, color=col, alpha=0.6, label=f'{fa}–{fb}')
                # binned trend
                r = stats.spearmanr(x, y)
                ax.annotate(f"$\\rho_s$={r.statistic:.2f}", xy=(0.02, 0.9 - 0.08 *
                            ['pythia-mamba', 'pythia-rwkv', 'mamba-rwkv'].index(f'{fa}-{fb}')),
                            xycoords='axes fraction', color=col, fontsize=8)
        ax.axhline(chance, color='k', ls='--', lw=0.8)
        ax.text(100, chance + 0.005, 'chance', fontsize=7)
        ax.set_xscale('log')
        ax.set_xlabel('min(params) of pair (M)')
        ax.set_ylabel('mutual $k$-NN alignment (max over layers)')
        ax.set_title('(c) PRH replication: representations align\nacross families, growing with scale')

        # (d) dissociation scatter: alignment vs capability gap
        ax = axes[3]
        xs, ys, cs = [], [], []
        for key, v in pairs.items():
            a, b = key.split('__')
            if a not in a1 or b not in a1:
                continue
            gap = abs(a1[a]['N16_d0.1']['acc'] - a1[b]['N16_d0.1']['acc'])
            xs.append(v['mutual_knn'])
            ys.append(gap)
            cs.append('#c0392b' if family(a) != family(b) else '#95a5a6')
        ax.scatter(xs, ys, c=cs, s=22, alpha=0.75)
        cross = [(x, y) for x, y, c in zip(xs, ys, cs) if c == '#c0392b']
        if len(cross) > 3:
            r = stats.spearmanr(*zip(*cross))
            ax.set_title(f'(d) alignment does not buy retrieval\n'
                         f'cross-family: $\\rho_s$={r.statistic:.2f} (p={r.pvalue:.2f})')
        ax.set_xlabel('representational alignment (mutual $k$-NN)')
        ax.set_ylabel('|retrieval accuracy gap| ($N{=}16$, d0.1)')
        from matplotlib.lines import Line2D
        pair_handles = [Line2D([], [], marker='o', ls='', color='#c0392b',
                               label='cross-family pair'),
                        Line2D([], [], marker='o', ls='', color='#95a5a6',
                               label='within-family pair')]

    fam_handles, fam_labels = axes[0].get_legend_handles_labels()
    all_h = fam_handles + (pair_handles if 'pair_handles' in dir() else [])
    fig.legend(all_h, [h.get_label() for h in all_h], loc='upper center',
               bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=8,
               columnspacing=1.8, handlelength=1.8)
    fig.suptitle('Exp A: representational convergence $\\neq$ capability convergence '
                 '(Pythia / Mamba / RWKV, all trained on The Pile)', y=1.03)
    fig.savefig(os.path.join(FD, 'expA_dissociation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FD, 'expA_dissociation.png'), dpi=150, bbox_inches='tight')
    print('saved', os.path.join(FD, 'expA_dissociation.pdf'))

    # verdict summary
    v = {}
    at = lambda m, cell: a1.get(m, {}).get(cell, {}).get('acc')
    big = {f: at(m, 'N16_d0.1') for f, m in
           dict(pythia='pythia-2.8b', mamba='mamba-2.8b-hf', rwkv='rwkv-4-3b-pile').items()}
    if all(x is not None for x in big.values()):
        v['A-ii: Pythia-2.8b exceeds Mamba-2.8b & RWKV-3b by >=30pp (N16 d0.1)'] = bool(
            big['pythia'] - max(big['mamba'], big['rwkv']) >= 0.30)
        v['acc_N16_d0.1'] = {k: round(x, 3) for k, x in big.items()}
    json.dump(v, open(os.path.join(RD, 'summary_expA.json'), 'w'), indent=1)
    print(v)


if __name__ == '__main__':
    main()
