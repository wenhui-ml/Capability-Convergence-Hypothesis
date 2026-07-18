"""Exp B figures + verdicts: measured version of paper Fig. 3 (fig3_fano_floor)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RD = os.path.join(os.path.dirname(__file__), '..', 'results', 'expB')
FD = os.path.join(os.path.dirname(__file__), '..', 'figs')
NG = [2, 4, 8, 16, 32, 64, 128]
B_BITS = 8
STATE_M = {'b_state64': 64, 'b_state256': 256, 'b_state1024': 1024}
COLORS = {'b_state64': '#c0392b', 'b_state256': '#e67e22', 'b_state1024': '#f1c40f',
          'b_swa': '#8e44ad', 'b_hybrid': '#27ae60', 'b_hybrid_local': '#16a085',
          'b_attn': '#2c3e50'}
LAB = {'b_state64': 'pure state $m{=}64$', 'b_state256': 'pure state $m{=}256$',
       'b_state1024': 'pure state $m{=}1024$', 'b_swa': 'sliding window $w{=}32$',
       'b_hybrid': 'hybrid (state$+$global)',
       'b_hybrid_local': 'state$+$local only', 'b_attn': 'full attention'}


def load():
    runs = {}
    for f in glob.glob(os.path.join(RD, '*.json')):
        d = json.load(open(f))
        if 'arch' not in d:          # skip summary_expB.json (written by this script)
            continue
        arch = d['arch'] + ('_mixedlayout' if d.get('layout', 'early') == 'mixed'
                            else '')
        key = (arch, d['kappa'], d['value_block'])
        runs.setdefault(key, []).append(d)
    return runs


def curve(runs, key, field='err', which='final'):
    if key not in runs:
        return None
    arr = np.array([[r[which][str(N)][field] for N in NG] for r in runs[key]])
    return arr


def n50(mean_err):
    """First N where error crosses 0.5 (log-linear interpolation)."""
    for i in range(len(NG) - 1):
        a, b = mean_err[i], mean_err[i + 1]
        if a < 0.5 <= b:
            f = (0.5 - a) / (b - a)
            return NG[i] * (NG[i + 1] / NG[i]) ** f
    return None


def below(ax, ncol):
    """Place the legend fully below the panel so it never overlaps the curves."""
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.19),
              ncol=ncol, frameon=False, columnspacing=1.1, handlelength=1.5,
              handletextpad=0.4, borderaxespad=0.0)


def main():
    runs = load()
    os.makedirs(FD, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    plt.subplots_adjust(hspace=0.62, wspace=0.28)

    # (a) headline scissors: error vs N, high kappa
    ax = axes[0, 0]
    for arch in ['b_state64', 'b_state256', 'b_state1024', 'b_swa',
                 'b_hybrid_local', 'b_hybrid', 'b_attn']:
        arr = curve(runs, (arch, 'high', 1))
        if arr is None:
            continue
        ax.plot(NG, arr.mean(0), 'o-', color=COLORS[arch], label=LAB[arch], lw=1.8, ms=4)
        ax.fill_between(NG, arr.min(0), arr.max(0), color=COLORS[arch], alpha=0.15)
    # amendment-7b arm: hybrid at doubled budget (display only; verdicts stay on
    # the frozen 12k aggregation above)
    f24 = sorted(glob.glob(os.path.join(RD, 'followup', 'b_hybrid_high_vb1_24k_s*.json')))
    if f24:
        a24 = np.array([[json.load(open(f))['final'][str(N)]['err'] for N in NG]
                        for f in f24])
        ax.plot(NG, a24.mean(0), 's--', color=COLORS['b_hybrid'], alpha=0.85,
                label='hybrid, $2\\times$ budget', lw=1.4, ms=4)
    ax.axhline(1 - 2 ** -B_BITS, color='k', ls='--', lw=0.8)
    ax.text(2.2, 0.965, 'chance', fontsize=7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(NG); ax.set_xticklabels(NG)
    ax.set_xlabel('planted bindings $N$  ($b{=}8$ bits each)')
    ax.set_ylabel('exact-retrieval error')
    ax.set_title('(a) the measured scissors gap ($\\kappa$ high)')
    below(ax, 2)

    # (b) load-ratio collapse with p_eff fit on m=256
    ax = axes[0, 1]
    m256 = curve(runs, ('b_state256', 'high', 1))
    p_eff = None
    if m256 is not None:
        N_half = n50(m256.mean(0))
        if N_half:
            p_eff = N_half * B_BITS / 256          # so x=1 at 50% crossing of m=256
            xs = np.linspace(0.05, 40, 400)
            ax.plot(xs, np.clip((1 - 1 / xs) - 1 / B_BITS, 0, 1), 'k--', lw=1.2,
                    label='Fano floor $(1{-}1/x)-1/b$')
            for arch, m in STATE_M.items():
                arr = curve(runs, (arch, 'high', 1))
                if arr is None:
                    continue
                x = np.array(NG) * B_BITS / (m * p_eff)
                ax.errorbar(x, arr.mean(0), yerr=[arr.mean(0) - arr.min(0),
                            arr.max(0) - arr.mean(0)], fmt='o-', color=COLORS[arch],
                            label=LAB[arch], lw=1.6, ms=4)
            hyb = curve(runs, ('b_hybrid', 'high', 1))
            if hyb is not None:
                x = np.array(NG) * B_BITS / (64 * p_eff)
                ax.plot(x, hyb.mean(0), 's-', color=COLORS['b_hybrid'],
                        label=LAB['b_hybrid'] + ' ($m{=}64$)', lw=1.6, ms=4)
            ax.set_xscale('log')
            ax.set_xlabel('load ratio $x = Nb/(m\\,p_{\\mathrm{eff}})$'
                          + f'   ($p_{{\\mathrm{{eff}}}}$={p_eff:.2f} bits/scalar)')
            ax.set_ylabel('exact-retrieval error')
            ax.set_title('(b) collapse onto one master curve')
            below(ax, 2)

    # (c) log-loss vs floor
    ax = axes[0, 2]
    if p_eff:
        xs = np.linspace(0.05, 40, 400)
        ax.plot(xs, np.clip(B_BITS * (1 - 1 / xs), 0, B_BITS), 'k--', lw=1.2,
                label='floor $b(1-1/x)$')
        for arch, m in STATE_M.items():
            arr = curve(runs, (arch, 'high', 1), field='nll_bits')
            if arr is None:
                continue
            x = np.array(NG) * B_BITS / (m * p_eff)
            ax.plot(x, arr.mean(0), 'o-', color=COLORS[arch], label=LAB[arch], lw=1.6, ms=4)
        hyb = curve(runs, ('b_hybrid', 'high', 1), field='nll_bits')
        if hyb is not None:
            ax.plot(np.array(NG) * B_BITS / (64 * p_eff), hyb.mean(0), 's-',
                    color=COLORS['b_hybrid'], label=LAB['b_hybrid'], lw=1.6, ms=4)
        ax.set_xscale('log')
        ax.set_xlabel('load ratio $x$')
        ax.set_ylabel('value log-loss (bits)')
        ax.set_title('(c) log-loss rides the information floor')
        below(ax, 2)

    # (d) kappa: semantic overlap — mixed-kappa-trained models (same weights,
    # both eval conditions); falls back to per-kappa-trained if absent
    ax = axes[1, 0]
    mixed = runs.get(('b_state256', 'mixed', 1))
    if mixed:
        for kappa, ls in [('high', '-'), ('low', ':')]:
            arr = np.array([[r['final'][kappa][str(N)]['err'] for N in NG]
                            for r in mixed])
            ax.plot(NG, arr.mean(0), 'o' + ls, color=COLORS['b_state256'],
                    label=f"state $m{{=}}256$ (mixed-trained), eval $\\kappa$ {kappa}",
                    lw=1.6, ms=4)
        ax.set_title('(d) $\\kappa$ interference at matched weights')
    else:
        for arch in ['b_state256', 'b_hybrid']:
            for kappa, ls in [('high', '-'), ('low', ':')]:
                arr = curve(runs, (arch, kappa, 1))
                if arr is None:
                    continue
                ax.plot(NG, arr.mean(0), 'o' + ls, color=COLORS[arch],
                        label=f"{LAB[arch]}, $\\kappa$ {kappa}", lw=1.6, ms=4)
        ax.set_title('(d) semantic-overlap kernel $\\kappa$ (per-$\\kappa$ trained)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(NG); ax.set_xticklabels(NG)
    ax.set_xlabel('$N$'); ax.set_ylabel('error')
    below(ax, 1)

    # (e) compressibility axis: error vs nominal and effective load
    ax = axes[1, 1]
    for vb, mk in [(1, 'o'), (4, 's'), (16, '^')]:
        arr = curve(runs, ('b_state256', 'high', vb))
        if arr is None:
            continue
        ax.plot(NG, arr.mean(0), mk + '-', color='#e67e22', alpha=1 - 0.25 * np.log2(vb) / 4,
                label=f'nominal, $c{{=}}{vb}$', lw=1.5, ms=4)
        ax.plot(np.array(NG) / vb, arr.mean(0), mk + ':', color='#2980b9',
                alpha=1 - 0.25 * np.log2(vb) / 4,
                label=f'effective, $c{{=}}{vb}$', lw=1.5, ms=4)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('$N$ (orange: nominal;  blue: effective $N/c$)')
    ax.set_ylabel('error')
    ax.set_title('(e) the wall bites on *effective* entropy')
    below(ax, 2)

    # (f) horizon wall: bindings early vs adjacent to query. Cliff shown on the
    # mixed-layout-trained SWA (sees both layouts in training).
    ax = axes[1, 2]
    f_archs = (['b_swa_mixedlayout'] if ('b_swa_mixedlayout', 'high', 1) in runs
               else []) + ['b_swa', 'b_state256']
    for arch in f_archs:
        for which, ls, mk in [('final', '-', 'o'), ('bindings_late', '--', 's')]:
            if (arch, 'high', 1) not in runs:
                continue
            arr = np.array([[r[which][str(N)]['err'] for N in NG]
                            for r in runs[(arch, 'high', 1)]])
            col = COLORS.get(arch, '#34495e')
            lab = LAB.get(arch, 'SWA $w{=}32$')
            ax.plot(NG, arr.mean(0), mk + ls, color=col,
                    label=f"{lab}, {'far' if which=='final' else 'near'}",
                    lw=1.6, ms=4)
    ax.set_xscale('log', base=2)
    ax.set_xticks(NG); ax.set_xticklabels(NG)
    ax.set_xlabel('$N$'); ax.set_ylabel('error')
    ax.set_title('(f) horizon wall: cliff, not slope')
    below(ax, 2)

    fig.suptitle('Exp B (P1): information floor and the scissors gap — measured '
                 '(NA$(N,b{=}8,B;\\kappa)$, $L{=}512$, 3 seeds)', y=0.995)
    fig.savefig(os.path.join(FD, 'expB_floor.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FD, 'expB_floor.png'), dpi=150, bbox_inches='tight')

    # ---- verdicts ----
    v, notes = {}, {}
    e64 = curve(runs, ('b_state64', 'high', 1))
    hyb = curve(runs, ('b_hybrid', 'high', 1))
    hyl = curve(runs, ('b_hybrid_local', 'high', 1))
    if e64 is not None:
        mono = all(e64.mean(0)[i] <= e64.mean(0)[i + 1] + 0.05 for i in range(len(NG) - 1))
        v['B-i: pure-state error rises with N toward saturation'] = bool(
            mono and e64.mean(0)[-1] > 0.5)
    if p_eff:
        notes['p_eff_bits_per_scalar'] = round(p_eff, 3)
        cross = {}
        for arch, m in STATE_M.items():
            arr = curve(runs, (arch, 'high', 1))
            if arr is not None:
                nh = n50(arr.mean(0))
                cross[arch] = round(nh * B_BITS / (m * p_eff), 3) if nh else None
        notes['x_at_50pct_error'] = cross
        vals = [x for x in cross.values() if x]
        v['B-ii: load-ratio collapse (all 50% crossings within [0.5,2]x of each other)'] = bool(
            len(vals) == 3 and max(vals) / min(vals) < 2.0)
    if hyb is not None and e64 is not None:
        v['B-iii: scissors (hybrid m=64 err<=0.05 all N; pure m=64 >0.5 at N=128)'] = bool(
            hyb.mean(0).max() <= 0.05 and e64.mean(0)[-1] > 0.5)
    if hyl is not None:
        v['B-viii: local-window hybrid is NOT rescued (err>0.5 at N=128)'] = bool(
            hyl.mean(0)[-1] > 0.5)
    swam = runs.get(('b_swa_mixedlayout', 'high', 1))
    if swam:
        near = np.array([[r['bindings_late'][str(N)]['err'] for N in [2, 4, 8]]
                         for r in swam])
        far = np.array([[r['final'][str(N)]['err'] for N in NG] for r in swam])
        v['B-iv (amended): horizon cliff (near N<=8 err<=0.2; far err>=0.9 all N)'] = bool(
            near.mean() <= 0.2 and far.mean(0).min() >= 0.9)
    # Fano sanity with nominal capacity 16 bits/scalar
    ok = True
    for arch, m in STATE_M.items():
        arr = curve(runs, (arch, 'high', 1))
        if arr is None:
            continue
        xnom = np.array(NG) * B_BITS / (m * 16)
        floor = np.clip((1 - 1 / xnom) - 1 / B_BITS, 0, 1)
        if (arr.mean(0) < floor - 0.02).any():
            ok = False
    v['B-v: no architecture beats the nominal-capacity Fano floor'] = bool(ok)
    mixed = runs.get(('b_state256', 'mixed', 1))
    if mixed:
        kh = np.array([[r['final']['high'][str(N)]['err'] for N in NG] for r in mixed])
        kl = np.array([[r['final']['low'][str(N)]['err'] for N in NG] for r in mixed])
        v['B-vi (amended): high-kappa >= low-kappa at matched weights'] = bool(
            kh.mean() >= kl.mean() - 0.02)
    else:
        ka = curve(runs, ('b_state256', 'high', 1))
        kb = curve(runs, ('b_state256', 'low', 1))
        if ka is not None and kb is not None:
            v['B-vi (as-designed, confounded): high >= low (per-kappa trained)'] = bool(
                ka.mean() >= kb.mean() - 0.02)
    json.dump(dict(verdicts=v, notes=notes),
              open(os.path.join(RD, 'summary_expB.json'), 'w'), indent=1)
    for k, x in v.items():
        print(('SUPPORTED' if x else 'FAILED'), '--', k)
    print(notes)


if __name__ == '__main__':
    main()
