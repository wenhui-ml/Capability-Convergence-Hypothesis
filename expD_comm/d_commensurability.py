"""Exp D (P3, miniature): channel commensurability in trained hybrids.

For each b_hybrid checkpoint (final = task-solving, early = failing, random init):
mutual-kNN alignment (k=10) between the state-channel block output (block 0,
DeltaNet) and the index-channel block output (block 1, global attention) at the
same (sequence, position) samples of the NA task, plus the checkpoint's task error.

Prediction D-i: solving hybrids align above the random-init floor; failing ones sit
near it (qualitative positive association — the paper's Fig. 4 claim).
"""
import argparse, glob, json, os, sys
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
    return mutual_knn(A, B), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(
        os.path.dirname(__file__), '..', 'results', 'expB'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda'
    specs, _ = arch_specs('b_hybrid')
    rows = []
    for ck in sorted(glob.glob(os.path.join(args.results_dir, 'b_hybrid_*_ckpt_*.pt'))):
        base = os.path.basename(ck)
        seed = int(base.split('_s')[1].split('_')[0])
        stage = 'final' if 'final' in base else 'early'
        model = SeqModel(VOCAB_B, VOCAB_B, 128, specs).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        aln, err = measure(model, device, seed)
        rows.append(dict(ckpt=base, stage=stage, seed=seed, align=aln, err=err))
        print(f"{base}: align={aln:.4f} err={err:.3f}", flush=True)
    # random-init floor (3 seeds)
    for seed in range(3):
        torch.manual_seed(900 + seed)
        model = SeqModel(VOCAB_B, VOCAB_B, 128, specs).to(device).eval()
        aln, err = measure(model, device, seed)
        rows.append(dict(ckpt=f'random_s{seed}', stage='random', seed=seed,
                         align=aln, err=err))
        print(f"random_s{seed}: align={aln:.4f} err={err:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(k=K, rows=rows), open(args.out, 'w'), indent=1)
    print("DONE", args.out)


if __name__ == '__main__':
    main()
