"""Exp A2: cross-family representational alignment on shared Pile text.

For each model: features = mean-pooled hidden state per snippet, per layer
(1024 snippets x 128 tokens from NeelNanda/pile-10k, identical token streams —
shared NeoX tokenizer). Saved to disk; alignment computed pairwise afterwards.

Metrics (PRH's): mutual-kNN overlap (k=10, cosine) and linear CKA, reported as
max over layer pairs (layers subsampled to <=16 per model, PRH-style sweep).

Stage 1 (per model): python a2_alignment.py --model X --stage features
Stage 2 (once):      python a2_alignment.py --stage align
"""
import argparse, glob, itertools, json, os, time
import numpy as np
import torch

FEAT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'expA', 'features')
N_SNIP, SNIP_LEN, K = 1024, 128, 10


def get_snippets(tok):
    from datasets import load_dataset
    docs = load_dataset("NeelNanda/pile-10k", split="train")
    snips, i = [], 0
    for d in docs:
        ids = tok(d["text"], add_special_tokens=False).input_ids
        if len(ids) >= SNIP_LEN:
            snips.append(ids[:SNIP_LEN])
        if len(snips) == N_SNIP:
            break
    return torch.tensor(snips)


@torch.no_grad()
def extract(model_name, out_path, bs=32):
    if os.path.exists(out_path):
        print('exists, skip:', out_path)
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    snips = get_snippets(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, output_hidden_states=True).to('cuda').eval()
    feats = None
    for i in range(0, N_SNIP, bs):
        ids = snips[i:i + bs].to('cuda')
        hs = model(input_ids=ids).hidden_states       # tuple (L+1) of (B,T,D)
        hs = torch.stack(hs, 0).float().mean(2)       # (L+1, B, D) mean over tokens
        if feats is None:
            feats = torch.zeros(hs.shape[0], N_SNIP, hs.shape[-1])
        feats[:, i:i + bs] = hs.cpu()
    torch.save(feats, out_path)
    print("saved", out_path, tuple(feats.shape), flush=True)


def subsample_layers(L, m=16):
    if L <= m:
        return list(range(L))
    return sorted(set(int(round(x)) for x in np.linspace(0, L - 1, m)))


def mutual_knn(A, B, k=K):
    """A,B: (n, d) torch on gpu. Return mean overlap of kNN sets (cosine)."""
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    sa = A @ A.T
    sb = B @ B.T
    sa.fill_diagonal_(-2)
    sb.fill_diagonal_(-2)
    ia = sa.topk(k, dim=1).indices
    ib = sb.topk(k, dim=1).indices
    n = A.shape[0]
    onehot_a = torch.zeros(n, n, device=A.device).scatter_(1, ia, 1)
    onehot_b = torch.zeros(n, n, device=A.device).scatter_(1, ib, 1)
    return ((onehot_a * onehot_b).sum(1) / k).mean().item()


def cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = (A.T @ B).norm() ** 2
    return (hsic / ((A.T @ A).norm() * (B.T @ B).norm())).item()


def align_all():
    files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.pt')))
    names = [os.path.basename(f)[:-3] for f in files]
    print("models:", names)
    results = {}
    for (fi, ni), (fj, nj) in itertools.combinations(zip(files, names), 2):
        Fi = torch.load(fi)
        Fj = torch.load(fj)
        li = subsample_layers(Fi.shape[0])
        lj = subsample_layers(Fj.shape[0])
        best_knn, best_cka = 0.0, 0.0
        for a in li:
            Xa = Fi[a].cuda()
            for b in lj:
                Xb = Fj[b].cuda()
                best_knn = max(best_knn, mutual_knn(Xa, Xb))
                best_cka = max(best_cka, cka(Xa, Xb))
        results[f"{ni}__{nj}"] = dict(mutual_knn=best_knn, cka=best_cka)
        print(f"{ni} vs {nj}: knn={best_knn:.4f} cka={best_cka:.4f}", flush=True)
    out = os.path.join(FEAT_DIR, '..', 'a2_alignment.json')
    json.dump(dict(k=K, n_snippets=N_SNIP, chance_knn=K / (N_SNIP - 1),
                   pairs=results), open(out, 'w'), indent=1)
    print("DONE", out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['features', 'align'], required=True)
    ap.add_argument('--model')
    ap.add_argument('--bs', type=int, default=32)
    args = ap.parse_args()
    os.makedirs(FEAT_DIR, exist_ok=True)
    if args.stage == 'features':
        short = args.model.split('/')[-1]
        extract(args.model, os.path.join(FEAT_DIR, f"{short}.pt"), args.bs)
    else:
        align_all()
