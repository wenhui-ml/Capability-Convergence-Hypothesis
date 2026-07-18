"""Exp B (P1): information floor / scissors gap — executable NA(N, b, B; kappa).

Train on mixed N (uniform over grid), evaluate exact-retrieval error and value
log-loss per N. Loss/metrics only at the answer position (last token predicts value).

Usage: python train_floor.py --arch b_state256 --kappa high --seed 0 --out r.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.models import SeqModel, arch_specs
from common.tasks import gen_na_batch, VOCAB_B, PAD

N_GRID = [2, 4, 8, 16, 32, 64, 128]
L = 512


def eval_grid(collide):
    """N values that fit L=512 given collide+1 bindings per key (amendment 8)."""
    return [N for N in N_GRID if 3 * N * (collide + 1) + 4 <= L]


def evaluate(model, kappa, device, seed, value_block=1, bindings_late=False,
             n=1024, bs=256, collide=0):
    model.eval()
    out = {}
    with torch.no_grad():
        for N in eval_grid(collide):
            rng = np.random.default_rng(777000 + seed * 100 + N)
            err, nll = 0, 0.0
            for _ in range(n // bs):
                toks, ans = gen_na_batch(bs, N, L, kappa, rng, device,
                                         value_block=value_block,
                                         bindings_late=bindings_late,
                                         collide=collide)
                logits = model(toks)[:, -1]
                err += (logits.argmax(-1) != ans).sum().item()
                nll += F.cross_entropy(logits, ans, reduction='sum').item()
            out[str(N)] = dict(err=err / n, nll_bits=nll / n / np.log(2))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--kappa', choices=['high', 'low', 'mixed'], default='high')
    ap.add_argument('--layout', choices=['early', 'mixed'], default='early',
                    help='mixed: bindings randomly early or adjacent-to-query '
                         'during training (horizon-cliff control)')
    ap.add_argument('--value_block', type=int, default=1)
    ap.add_argument('--collide', type=int, default=0,
                    help='amendment 8: colliding rebinds per key (Assumption-sep '
                         'negative control); Bayes floor err=c/(c+1)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=7e-4)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda'

    specs, desc = arch_specs(args.arch, args.d_model)
    model = SeqModel(VOCAB_B, VOCAB_B, args.d_model, specs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)

    t0, hist = time.time(), []
    train_grid = eval_grid(args.collide)
    for step in range(1, args.steps + 1):
        N = int(rng.choice(train_grid))
        kap = str(rng.choice(['high', 'low'])) if args.kappa == 'mixed' else args.kappa
        late = bool(rng.integers(0, 2)) if args.layout == 'mixed' else False
        toks, ans = gen_na_batch(args.bs, N, L, kap, rng, device,
                                 value_block=args.value_block, bindings_late=late,
                                 collide=args.collide)
        logits = model(toks)
        # auxiliary LM loss over the stream (dense supervision; teaches the
        # key->value binding mechanism) + answer loss. Metrics stay answer-only.
        tgt = toks[:, 1:].clone()
        tgt[tgt == PAD] = -100
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_B), tgt.reshape(-1),
                             ignore_index=-100)
        loss = lm + F.cross_entropy(logits[:, -1], ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 400 and args.arch.startswith('b_hybrid'):
            torch.save(model.state_dict(), args.out.replace('.json', '_ckpt_early.pt'))
        if step % 2000 == 0 or step == args.steps:
            if args.kappa == 'mixed':
                ev = {k: evaluate(model, k, device, args.seed,
                                  value_block=args.value_block)
                      for k in ['high', 'low']}
            else:
                ev = evaluate(model, args.kappa, device, args.seed,
                              value_block=args.value_block, collide=args.collide)
            hist.append(dict(step=step, loss=loss.item(), eval=ev))
            disp = ev['high'] if args.kappa == 'mixed' else ev
            print(f"[{args.arch}/{args.kappa}/vb{args.value_block}"
                  f"/cx{args.collide}/s{args.seed}] "
                  f"step {step} loss {loss.item():.4f} "
                  f"err {[round(disp[str(N)]['err'],3) for N in train_grid]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # horizon-wall probe: bindings adjacent to query (inside window)
    late_kappa = 'high' if args.kappa == 'mixed' else args.kappa
    late = evaluate(model, late_kappa, device, args.seed,
                    value_block=args.value_block, bindings_late=True,
                    collide=args.collide)
    if args.arch.startswith('b_hybrid'):
        torch.save(model.state_dict(), args.out.replace('.json', '_ckpt_final.pt'))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    c = args.collide
    json.dump(dict(arch=args.arch, desc=desc, kappa=args.kappa, layout=args.layout,
                   value_block=args.value_block, collide=c,
                   bayes_floor=dict(err=c / (c + 1), nll_bits=float(np.log2(c + 1))),
                   seed=args.seed, lr=args.lr,
                   steps=args.steps, n_params=model.n_params(),
                   final=hist[-1]['eval'], bindings_late=late, history=hist,
                   wall_seconds=round(time.time() - t0)),
              open(args.out, 'w'), indent=1)
    print("DONE", args.out, flush=True)


if __name__ == '__main__':
    main()
