"""Exp E figure (expE_witness.pdf): the conjunction witness, measured.

LEFT  -- accuracy vs load N at the long-extrapolation length T=256, all five arms
         (per-seed dots + mean lines; pass line 0.9). The recurrent-front hybrid passes
         the registered cell (T=256, N=16) on every seed while no other arm meets the
         2/3-seed criterion; at the supra-capacity load N=64 the hybrid's worst seed
         (0.706) still exceeds the pure arm's best (0.654) -- a graded margin.
RIGHT -- D-ii channel complementarity (P3'): silencing either channel collapses the
         registered cell (irreducibly two-channel), and the answer is linearly decodable
         from the index mixer (1.0) but not the state mixer (~0.004): retrieval localizes
         to the index channel exactly as the mechanism predicts.

Reads results/expE/{summary_expE.json, d2_complementarity.json}. CPU-only.
Usage:  python plot_e.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RD = os.path.join(HERE, "..", "results", "expE")
FD = os.path.join(HERE, "..", "figs")
NS = [4, 16, 64]

ARMS = [  # (key, label, color, marker)
    ("e_hybrid_d2a",       "recurrent-front hybrid (state+global)", "#27ae60", "o"),
    ("e_state3",           "pure recurrence",                       "#e67e22", "s"),
    ("e_hybrid_local_d2a", "window-32 hybrid",                      "#8e44ad", "^"),
    ("e_hybrid_b1_d2a",    "$\\beta\\in(0,1)$ hybrid",              "#c0392b", "D"),
    ("e_attn8",            "8-layer Transformer (attn-only)",       "#2c3e50", "v"),
]


def main():
    pa = json.load(open(os.path.join(RD, "summary_expE.json")))["per_arm"]
    d2 = json.load(open(os.path.join(RD, "d2_complementarity.json")))["rows"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---------- LEFT: accuracy vs N at T=256 ----------
    x = np.arange(len(NS))
    for key, lab, col, mk in ARMS:
        cells = [pa[key][f"T256_N{N}"] for N in NS]
        means = [c["mean"] for c in cells]
        axL.plot(x, means, mk + "-", color=col, lw=1.9, ms=6, label=lab, zorder=3)
        for xi, c in zip(x, cells):
            axL.scatter([xi] * len(c["per_seed"]), c["per_seed"], s=14, color=col,
                        alpha=0.35, zorder=2, edgecolors="none")
    axL.axhline(0.9, ls="--", color="k", lw=0.9)
    axL.text(0.02, 0.915, "pass line = 0.9", fontsize=7.2, color="k")
    # Two callouts. The orange (pure) curve descends diagonally across the mid band, so
    # both labels are placed high, in the wedge that stays clear of every curve: above
    # the orange curve and below the green curve on the right half of the panel. Each
    # leader points down to its data (green N=16 point; the N=64 crossover dots).
    axL.annotate("registered cell ($N{=}16$):\nonly the hybrid passes",
                 xy=(1.0, 0.965), xytext=(1.14, 0.74),
                 fontsize=7.0, color="#1e8449", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#1e8449", lw=0.8,
                                 connectionstyle="arc3,rad=-0.15"))
    axL.annotate("supra-capacity load ($N{=}64$):\nhybrid worst 0.71 $>$ pure best 0.65",
                 xy=(1.98, 0.69), xytext=(1.30, 0.52),
                 fontsize=7.0, color="#444", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#444", lw=0.8,
                                 connectionstyle="arc3,rad=0.25"))
    axL.set_xticks(x); axL.set_xticklabels(NS)
    axL.set_xlabel("load $N$ (bindings) at extrapolation length $T{=}256$")
    axL.set_ylabel("composite-task accuracy")
    axL.set_ylim(-0.03, 1.05); axL.set_xlim(-0.25, 2.25)
    axL.grid(True, axis="y", lw=0.4, alpha=0.3)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.set_title("(a) the conjunction separates only the hybrid", fontsize=9.5)
    axL.legend(fontsize=6.9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
               frameon=False, columnspacing=1.2, handlelength=1.6, handletextpad=0.4)

    # ---------- RIGHT: D-ii complementarity ----------
    base = np.mean([r["base"]["T256_N16"] for r in d2])
    les_s = np.mean([r["lesion"]["state"]["T256_N16"] for r in d2])
    les_a = np.mean([r["lesion"]["attn"]["T256_N16"] for r in d2])
    dec_idx = np.mean([r["probes"]["attn_to_value"] for r in d2])
    dec_st = np.mean([r["probes"]["state_to_value"] for r in d2])

    labels = ["full\nhybrid", "$-$state\nchannel", "$-$index\nchannel",
              "from\nindex", "from\nstate"]
    vals = [base, les_s, les_a, dec_idx, dec_st]
    cols = ["#27ae60", "#c0392b", "#c0392b", "#2c3e50", "#e67e22"]
    xb = [0, 1, 2, 3.6, 4.6]
    axR.bar(xb, vals, color=cols, width=0.72, zorder=3, edgecolor="white", lw=0.6)
    for xi, v in zip(xb, vals):
        axR.text(xi, v + 0.03, f"{v:.2f}" if v >= 0.01 else f"{v:.3f}",
                 ha="center", fontsize=7.2, color="#333")
    axR.set_xticks(xb); axR.set_xticklabels(labels, fontsize=7.4)
    axR.set_ylim(0, 1.12)
    axR.set_ylabel("accuracy  /  linear decodability")
    axR.grid(True, axis="y", lw=0.4, alpha=0.3)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    # group dividers / captions
    axR.axvline(2.8, color="#bbb", lw=0.8, ls=":")
    axR.text(1.0, 1.07, "cell accuracy under channel lesion", fontsize=6.9,
             ha="center", color="#555")
    axR.text(4.1, 1.07, "answer decodable", fontsize=6.9, ha="center", color="#555")
    axR.set_title("(b) D-ii: irreducibly two-channel; retrieval on the index", fontsize=9.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FD, f"expE_witness.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done: expE_witness.pdf  (base %.3f, lesion s/a %.3f/%.3f, decode idx/st %.2f/%.3f)"
          % (base, les_s, les_a, dec_idx, dec_st))


if __name__ == "__main__":
    main()
