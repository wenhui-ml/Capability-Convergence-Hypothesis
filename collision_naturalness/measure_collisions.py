"""Exploratory (post-hoc, NOT pre-registered) first measurement of the write-time
key-separability protocol proposed in the paper's call to action (Assumption 1
naturalness, Limitation iii).

Operationalization (surface-form proxy, v1):
  - A "window" is one write context: a Pile document (natural text) or a Python
    source file (code corpus).
  - Keys per window:
      * text entities : capitalized 1-4 word spans (len>=4, not sentence-initial)
      * numeric IDs   : \\d{4,}
      * code idents   : ast-extracted def/class/assignment names (len>=4)
  - Near-collision between two DISTINCT keys in the same window:
      * strings   : character-trigram Jaccard >= 0.5
      * numeric   : same length and Hamming distance <= 1
  - Collision rate = fraction of distinct keys with >=1 near-neighbor in-window;
    separable fraction = 1 - collision rate.
  Caveats (stated in the paper): surface-form separability is a first-order proxy;
  it sees neither semantic aliasing (different surface, same referent) nor
  ambiguity (same surface, different referent). Exact-duplicate mentions are not
  counted as collisions (re-mentioning one entity is not a distinct binding).

Usage: python measure_collisions.py --out results.json
"""
import argparse, ast, glob, json, os, re, sys
import numpy as np

CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,3})\b")
NUM = re.compile(r"\b\d{4,}\b")


def trigrams(s):
    s = s.lower()
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def jaccard(a, b):
    return len(a & b) / len(a | b)


MAX_KEYS = 1000  # per-window cap (uniform random sample, fixed seed)
_rng = np.random.default_rng(1234)


def near_string(keys):
    """Fraction of distinct keys with a trigram-Jaccard>=0.5 neighbor in-window.
    Exact within the (possibly capped) window, via an inverted trigram index:
    shared-trigram counts give |A∩B| directly, so Jaccard = c/(|A|+|B|-c)."""
    ks = list(keys)
    if len(ks) > MAX_KEYS:
        ks = list(_rng.choice(ks, MAX_KEYS, replace=False))
    grams = [trigrams(k) for k in ks]
    posting = {}
    for i, g in enumerate(grams):
        for t in g:
            posting.setdefault(t, []).append(i)
    hit = np.zeros(len(ks), bool)
    for i, g in enumerate(grams):
        shared = {}
        for t in g:
            for j in posting[t]:
                if j > i:
                    shared[j] = shared.get(j, 0) + 1
        for j, c in shared.items():
            if c / (len(g) + len(grams[j]) - c) >= 0.5:
                hit[i] = hit[j] = True
    return hit


def near_numeric(keys):
    ks = list(keys)
    hit = np.zeros(len(ks), bool)
    by_len = {}
    for i, k in enumerate(ks):
        by_len.setdefault(len(k), []).append(i)
    for idxs in by_len.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if sum(x != y for x, y in zip(ks[i], ks[j])) <= 1:
                    hit[i] = hit[j] = True
    return hit


def stats_for(windows, near_fn):
    """windows: list of sets of distinct keys."""
    per_key_hits, n_keys, per_win = 0, 0, []
    sizes = []
    for keys in windows:
        if len(keys) < 2:
            continue
        hit = near_fn(keys)
        per_key_hits += int(hit.sum())
        n_keys += len(keys)
        per_win.append(hit.mean())
        sizes.append(len(keys))
    return dict(n_windows=len(per_win),
                mean_keys_per_window=float(np.mean(sizes)),
                collision_rate=per_key_hits / max(n_keys, 1),
                separable_fraction=1 - per_key_hits / max(n_keys, 1),
                per_window_mean_rate=float(np.mean(per_win)))


def text_windows(docs, pool=1):
    ents, nums = [], []
    buf_e, buf_n = set(), set()
    for i, t in enumerate(docs):
        buf_e |= {m.group(1) for m in CAP.finditer(t) if len(m.group(1)) >= 4}
        buf_n |= set(NUM.findall(t))
        if (i + 1) % pool == 0:
            ents.append(buf_e); nums.append(buf_n)
            buf_e, buf_n = set(), set()
    return ents, nums


def code_windows(files, pool=1):
    wins, buf = [], set()
    for i, f in enumerate(files):
        try:
            src = open(f, encoding='utf-8', errors='ignore').read()
            tree = ast.parse(src)
        except Exception:
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        buf |= {n for n in names if len(n) >= 4 and not n.startswith('__')}
        if (i + 1) % pool == 0:
            wins.append(buf); buf = set()
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_collisions.json')
    ap.add_argument('--n_docs', type=int, default=10000)
    ap.add_argument('--n_files', type=int, default=3000)
    args = ap.parse_args()

    from datasets import load_dataset
    docs = [r['text'] for r in load_dataset('NeelNanda/pile-10k', split='train')][:args.n_docs]

    import numpy as _np, scipy as _sp
    roots = [os.path.dirname(_np.__file__), os.path.dirname(_sp.__file__)]
    files = sorted(f for r in roots for f in glob.glob(os.path.join(r, '**/*.py'), recursive=True))
    rng = np.random.default_rng(0)
    files = list(rng.permutation(files))[:args.n_files]

    out = {'corpus': {'text': 'NeelNanda/pile-10k (the Exp A snippet source)',
                      'code': f'{len(files)} .py files from numpy+scipy source trees'},
           'params': {'jaccard_threshold': 0.5, 'max_keys_per_window': MAX_KEYS}}
    for pool, tag in [(1, 'doc'), (8, '8doc')]:
        ents, nums = text_windows(docs, pool)
        out[f'text_entities_{tag}'] = stats_for(ents, near_string)
        print(f'done text_entities_{tag}', flush=True)
        out[f'text_numeric_ids_{tag}'] = stats_for(nums, near_numeric)
        print(f'done text_numeric_ids_{tag}', flush=True)
    for pool, tag in [(1, 'file'), (8, '8file')]:
        wins = code_windows(files, pool)
        out[f'code_identifiers_{tag}'] = stats_for(wins, near_string)
        print(f'done code_identifiers_{tag}', flush=True)

    json.dump(out, open(args.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
