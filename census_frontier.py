"""Reproducible census statistics for R1 (Section 4.1 / Appendix census).

Recomputes, from the single source of truth figs_en/census_data.json:
  (1) the full-set statistics over all hybrids with rho in (0,1)  -- the headline R1 numbers;
  (2) the enrichment ratios against three nulls (uniform / design-menu / scale-invariant);
  (3) a PRODUCTION-FRONTIER subset under an explicit, inspectable rule, with the exact
      resulting median / sd / in-band fraction.

The frontier rule is stated mechanically so the subset is reproducible rather than
hand-curated. It matches the paper's prose exclusions ("excluding distillation-ablation
sweeps and small edge models"):
  - distillation conversions: Transformer->SSM distillations (name contains InLlama /
    -distill / Mamba-Llama, plus the academic conversions M1 and Hybrid-Phi-Mamba);
  - on-device edge line: Liquid AI LFM2 / LFM2.5 family.
Single-shared-cache production designs (YOCO, SambaY, Nemotron-H) are KEPT: they are
production models that retain genuine global access, merely below-band.

Usage:  python census_frontier.py
"""
import json
import os
import re
import numpy as np

HERE = os.path.dirname(__file__)
CENSUS = os.path.join(HERE, "..", "figs_en", "census_data.json")
BAND = (1.0 / 12.0, 1.0 / 4.0)


def load_hybrids():
    d = json.load(open(CENSUS))
    recs = d if isinstance(d, list) else list(d.values())
    return [r for r in recs if r.get("rho") not in (None, "") and 0 < float(r["rho"]) < 1]


def in_band(x):
    return BAND[0] <= x <= BAND[1]


def is_distillation(r):
    n = r.get("name", "")
    return bool(re.search(r"InLlama|-distill|Mamba-?Llama|Mamba2-Llama|^M1-|Hybrid-Phi-Mamba",
                          n, re.I))


def is_edge(r):
    n, org = r.get("name", ""), r.get("organization", "")
    return n.startswith("LFM2") or "Liquid AI" in org


def stats(rhos, label):
    a = np.asarray(rhos, float)
    frac = np.mean([in_band(x) for x in a])
    print(f"[{label}]  n={len(a)}  mean={a.mean():.4f}  median={np.median(a):.4f}  "
          f"sd_pop={a.std():.4f}  range=[{a.min():.4f},{a.max():.4f}]  "
          f"in-band={frac:.4f} ({int(round(frac*len(a)))}/{len(a)})")
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                sd=float(a.std()), lo=float(a.min()), hi=float(a.max()), in_band=float(frac))


def enrichments(full):
    """Enrichment of the in-band fraction over three nulls."""
    band_w = BAND[1] - BAND[0]                       # 1/6 of the (0,1) axis
    f = full["in_band"]
    # uniform null on (0,1)
    uni = f / band_w
    # design-menu null: uniform over reduced fractions p/q with q<=Q
    from math import gcd
    def menu(Q):  # uniform over *reduced* fractions p/q, gcd(p,q)=1, q<=Q
        fr = {p / q for q in range(2, Q + 1) for p in range(1, q) if gcd(p, q) == 1}
        m = np.mean([in_band(x) for x in fr])
        return f / m
    # scale-invariant (log-uniform) null on the observed hybrid range
    lo, hi = full["lo"], full["hi"]
    li = (np.log(BAND[1]) - np.log(BAND[0])) / (np.log(hi) - np.log(lo))
    print(f"enrichment vs uniform null            : {uni:.2f}x")
    print(f"enrichment vs design-menu null (q<=12): {menu(12):.2f}x")
    print(f"enrichment vs design-menu null (q<=32): {menu(32):.2f}x")
    print(f"enrichment vs scale-invariant null    : {f/li:.2f}x  (null mass {li:.3f})")


def main():
    hy = load_hybrids()
    rhos = [float(r["rho"]) for r in hy]
    print("=" * 78)
    full = stats(rhos, "ALL hybrids rho in (0,1)")
    enrichments(full)
    print("=" * 78)

    dropped_d = [r["name"] for r in hy if is_distillation(r)]
    dropped_e = [r["name"] for r in hy if is_edge(r) and not is_distillation(r)]
    frontier = [float(r["rho"]) for r in hy
                if not is_distillation(r) and not is_edge(r)]
    print(f"dropped as distillation conversions ({len(dropped_d)}): {dropped_d}")
    print(f"dropped as edge line ({len(dropped_e)}): {dropped_e}")
    stats(frontier, "PRODUCTION-FRONTIER subset")
    print("=" * 78)

    # per-cohort dispersion by release year
    print("per-cohort dispersion (population sd of rho by year):")
    for yr in (2024, 2025, 2026):
        c = [float(r["rho"]) for r in hy
             if int(float(r.get("year_dec", 0))) == yr]
        if c:
            print(f"  {yr}: n={len(c)}  sd={np.std(c):.4f}")


if __name__ == "__main__":
    main()
