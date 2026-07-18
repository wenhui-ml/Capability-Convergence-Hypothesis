"""Exp D v2 (P3 miniature): protocol-completion re-measurement (amendment 7d).

Changes over d_commensurability.py (whose output d_comm.json stays frozen):
1. Arch is inferred from the checkpoint filename, so b_hybrid_local checkpoints
   are measured under their true sliding-window spec (v1 loaded them with the
   global-attention spec: identical state-dict keys, wrong runtime mask).
2. Adds the pre-registered failing arm b_hybrid_b1 (beta in (0,1) state channel).
3. Adds linear CKA as a post-hoc secondary metric next to mutual-kNN (k=10).
4. Sweeps results/expB plus results/expB/followup (24k-budget hybrids tagged by
   the _24k_ filename marker).
Random-init floors are computed per arch variant (3 seeds each). The probe rng
matches v1 exactly, so v1/v2 mutual-kNN numbers are directly comparable.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.models import SeqModel, arch_specs
from common.tasks import gen_na_batch, VOCAB_B, PAD

K = 10


def mutual_knn(A, B, k=K):
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    sa, sb = A @ A.T, B @ B.T
    sa.fill_diagonal_(-2); sb.fill_diagonal_(-2)
    ia = sa.topk(k, 1).indices
    ib = sb.topk(k, 1).indices
    n = A.shape[0]
    oa = torch.zeros(n, n, device=A.device).scatter_(1, ia, 1)
    ob = torch.zeros(n, n, device=A.device).scatter_(1, ib, 1)
    return ((oa * ob).sum(1) / k).mean().item()


def linear_cka(A, B):
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    num = (A.T @ B).norm() ** 2
    den = (A.T @ A).norm() * (B.T @ B).norm()
    return (num / den).item() if den > 0 else float('nan')


@torch.no_grad()
def measure(model, device, seed, N=32, n_seq=64, n_pos=1024):
    rng = np.random.default_rng(555000 + seed)
    toks, ans = gen_na_batch(n_seq, N, 512, 'high', rng, device)
    logits, outs = model(toks, return_layer_outputs=True)
    err = (logits[:, -1].argmax(-1) != ans).float().mean().item()
    state_f, attn_f = outs[0], outs[1]                    # (B,T,D)
    mask = (toks != PAD)
    idx = mask.reshape(-1).nonzero().squeeze(1)
    sel = idx[torch.randperm(len(idx), device=device)[:n_pos]]
    A = state_f.reshape(-1, state_f.shape[-1])[sel].float()
    B = attn_f.reshape(-1, attn_f.shape[-1])[sel].float()
    return mutual_knn(A, B), linear_cka(A, B), err


def arch_of(base):
    if base.startswith('b_hybrid_local'):
        return 'b_hybrid_local'
    if base.startswith('b_hybrid_b1'):
        return 'b_hybrid_b1'
    return 'b_hybrid'


def main():
    ap = argparse.ArgumentParser()
    rd = os.path.join(os.path.dirname(__file__), '..', 'results', 'expB')
    ap.add_argument('--results_dirs', nargs='+',
                    default=[rd, os.path.join(rd, 'followup')])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda'
    rows = []
    cks = sorted(c for d in args.results_dirs
                 for c in glob.glob(os.path.join(d, 'b_hybrid*_ckpt_*.pt')))
    for ck in cks:
        base = os.path.basename(ck)
        arch = arch_of(base)
        seed = int(re.search(r'_s(\d+)_ckpt', base).group(1))
        stage = 'final' if 'final' in base else 'early'
        budget = '24k' if '_24k_' in base else '12k'
        specs, _ = arch_specs(arch)
        model = SeqModel(VOCAB_B, VOCAB_B, 128, specs).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        knn, cka, err = measure(model, device, seed)
        rows.append(dict(ckpt=base, arch=arch, stage=stage, seed=seed,
                         budget=budget, align_knn=knn, align_cka=cka, err=err))
        print(f"{base} [{arch}/{budget}]: knn={knn:.4f} cka={cka:.4f} "
              f"err={err:.3f}", flush=True)
    # random-init floor, 3 seeds per arch variant
    for arch in ['b_hybrid', 'b_hybrid_local', 'b_hybrid_b1']:
        specs, _ = arch_specs(arch)
        for seed in range(3):
            torch.manual_seed(900 + seed)
            model = SeqModel(VOCAB_B, VOCAB_B, 128, specs).to(device).eval()
            knn, cka, err = measure(model, device, seed)
            rows.append(dict(ckpt=f'random_{arch}_s{seed}', arch=arch,
                             stage='random', seed=seed, budget=None,
                             align_knn=knn, align_cka=cka, err=err))
            print(f"random_{arch}_s{seed}: knn={knn:.4f} cka={cka:.4f} "
                  f"err={err:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(k=K, rows=rows), open(args.out, 'w'), indent=1)
    print("DONE", args.out)


if __name__ == '__main__':
    main()
