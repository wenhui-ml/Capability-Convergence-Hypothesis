"""Exp E (amendment 9): the composite witness NA-composed-with-S5.

Bindings [perm-address -> 8-bit value] written first; a swap-instruction stream
then drives the referent through S5; the query asks for the value bound to the
COMPOSED final referent. One prediction requires state tracking AND exact
retrieval — the measured form of Prop. nonpreserve's strict inclusion.

Train: N uniform on {4,8,16,32,64}, T uniform in [3,40]. Eval: acc at
T in {40,64,128,256} x N in {4,16,64}, 1024 instances/cell.
Usage: python train_comp.py --arch e_hybrid --seed 0 --out r.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.models import SeqModel, arch_specs
from common.tasks import gen_comp_batch, VOCAB_E, PAD, E_P0

N_TRAIN = [4, 8, 16, 32, 64]
T_EVAL = [40, 64, 128, 256]
N_EVAL = [4, 16, 64]
L = 512


def evaluate(model, device, seed, n=1024, bs=256, t_eval=None):
    model.eval()
    out = {}
    with torch.no_grad():
        for T in (t_eval or T_EVAL):
            for N in N_EVAL:
                rng = np.random.default_rng(888000 + seed * 1000 + T * 10 + N)
                acc, aacc, npos = 0, 0, 0
                for _ in range(n // bs):
                    toks, ans, aux = gen_comp_batch(bs, N, T, L, rng, device,
                                                    return_states=True)
                    logits = model(toks)
                    acc += (logits[:, -1].argmax(-1) == ans).sum().item()
                    # running-answer diagnostic over all instruction positions
                    pred = logits[:, L - 1 - T:L - 1].argmax(-1)
                    aacc += (pred == aux).sum().item()
                    npos += aux.numel()
                out[f"T{T}_N{N}"] = acc / n
                out[f"aux_T{T}_N{N}"] = aacc / npos
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=7e-4)
    ap.add_argument('--answer_w', type=float, default=4.0)
    ap.add_argument('--tie', type=int, default=0,
                    help='weight tying (pilot round 3 showed it harms; default off)')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda'
    # e_attn8 reuses the depth-advantaged Transformer control spec from Exp C
    specs, desc = arch_specs('c_attn8' if args.arch == 'e_attn8' else args.arch,
                             args.d_model)
    model = SeqModel(VOCAB_E, VOCAB_E, args.d_model, specs,
                     tie_embeddings=bool(args.tie)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)

    t0, hist = time.time(), []
    for step in range(1, args.steps + 1):
        N = int(rng.choice(N_TRAIN))
        T = int(rng.integers(3, 41))
        toks, ans, aux = gen_comp_batch(args.bs, N, T, L, rng, device,
                                        return_states=True)
        logits = model(toks)
        # amendment 9 supervision, uniform across arms: (a) LM over the stream
        # (amendment 4 precedent — teaches the binding mechanism), (b) dense
        # "running answer" labels at instruction positions (the composite
        # analogue of Exp C's per-position labels — value bound to the current
        # referent when planted, else its address), (c) the weighted answer.
        tgt = toks[:, 1:].clone()
        tgt[tgt == PAD] = -100
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_E), tgt.reshape(-1),
                             ignore_index=-100)
        aux_logits = logits[:, L - 1 - T:L - 1].reshape(-1, VOCAB_E)
        track = F.cross_entropy(aux_logits, aux.reshape(-1))
        loss = lm + track + args.answer_w * F.cross_entropy(logits[:, -1], ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step == 400 and args.arch.startswith('e_hybrid'):
            torch.save(model.state_dict(), args.out.replace('.json', '_ckpt_early.pt'))
        if step % 3000 == 0 or step == args.steps:
            ev = evaluate(model, device, args.seed)
            hist.append(dict(step=step, loss=loss.item(), eval=ev))
            print(f"[{args.arch}/s{args.seed}] step {step} loss {loss.item():.4f} "
                  f"acc(T256) {[round(ev[f'T256_N{N}'], 3) for N in N_EVAL]} "
                  f"acc(T40) {[round(ev[f'T40_N{N}'], 3) for N in N_EVAL]} "
                  f"aux(T40/T256@N16) "
                  f"{round(ev['aux_T40_N16'], 3)}/{round(ev['aux_T256_N16'], 3)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if args.arch.startswith('e_hybrid') or args.arch == 'e_state':
        torch.save(model.state_dict(), args.out.replace('.json', '_ckpt_final.pt'))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(arch=args.arch, desc=desc, seed=args.seed, lr=args.lr,
                   steps=args.steps, n_params=model.n_params(),
                   final_acc=hist[-1]['eval'], history=hist,
                   wall_seconds=round(time.time() - t0)),
              open(args.out, 'w'), indent=1)
    print("DONE", args.out, flush=True)


if __name__ == '__main__':
    main()
