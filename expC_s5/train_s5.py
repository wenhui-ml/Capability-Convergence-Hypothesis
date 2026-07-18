"""Exp C (P2): S5 state-tracking bifurcation.

Train at T=40 (supervised at every position => covers all prefix lengths <= 40),
evaluate last-position accuracy at prefix lengths {40,64,128,256,512}.

Usage: python train_s5.py --arch c_delta_b1 --stream general --seed 0 --out results.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.models import SeqModel, arch_specs
from common.tasks import gen_s5_batch

EVAL_LENGTHS = [40, 64, 128, 256, 512]


def evaluate(model, stream, rng, device, n=2000, bs=250):
    model.eval()
    accs = {}
    correct = {T: 0 for T in EVAL_LENGTHS}
    with torch.no_grad():
        for _ in range(n // bs):
            toks, labels = gen_s5_batch(bs, max(EVAL_LENGTHS), stream, rng, device)
            logits = model(toks)
            pred = logits.argmax(-1)
            for T in EVAL_LENGTHS:
                correct[T] += (pred[:, T - 1] == labels[:, T - 1]).sum().item()
    for T in EVAL_LENGTHS:
        accs[str(T)] = correct[T] / n
    model.train()
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--stream', choices=['swap', 'general'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--train_T', type=int, default=40)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(10000 + args.seed)
    device = 'cuda'

    specs, desc = arch_specs(args.arch, args.d_model)
    model = SeqModel(120, 120, args.d_model, specs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)

    t0, hist = time.time(), []
    for step in range(1, args.steps + 1):
        toks, labels = gen_s5_batch(args.bs, args.train_T, args.stream, rng, device)
        logits = model(toks)
        loss = F.cross_entropy(logits.reshape(-1, 120), labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 2500 == 0 or step == args.steps:
            accs = evaluate(model, args.stream, np.random.default_rng(10000 + args.seed),
                            device)
            hist.append(dict(step=step, loss=loss.item(), acc=accs))
            print(f"[{args.arch}/{args.stream}/s{args.seed}] step {step} "
                  f"loss {loss.item():.4f} acc {accs} ({time.time()-t0:.0f}s)", flush=True)

    final = hist[-1]['acc']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(arch=args.arch, desc=desc, stream=args.stream, seed=args.seed,
                   lr=args.lr, steps=args.steps, n_params=model.n_params(),
                   final_acc=final, history=hist,
                   wall_seconds=round(time.time() - t0)),
              open(args.out, 'w'), indent=1)
    print("DONE", args.out, flush=True)


if __name__ == '__main__':
    main()
