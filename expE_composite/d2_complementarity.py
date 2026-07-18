"""Exp D-ii (amendment 10): channel complementarity on Exp E hybrid checkpoints.

(a) Decodability double dissociation — linear probes from each channel's MIXER
    output (residual-stream delta, isolating the channel's own contribution):
      state-mixer vs attn-mixer  ->  current referent r_k (120-way, at
      instruction positions where r_k is unplanted)  and  answer value
      (256-way, at the QUERY position).
    Frozen prediction: state decodes r_k better by >=20pp; attn decodes the
    value better by >=20pp; each channel's primary decode >= 0.5.
(b) Channel lesion — bypass one mixer (zero its output; the block reduces to
    x + MLP(LN(x))) and re-evaluate. Frozen prediction: state-lesion collapses
    (T=256,N=16) by >=50pp and more than it collapses (T=40,N=64); attn-lesion
    collapses (T=40,N=64) by >=50pp and more than (T=256,N=16).
Usage: python d2_complementarity.py --ckpts results/expE/e_hybrid_s*_ckpt_final.pt \
          --out results/expE/d2_complementarity.json
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.models import SeqModel, arch_specs, DeltaMixer, AttnMixer
from common.tasks import gen_comp_batch, VOCAB_E, E_P0, E_V0

L = 512


def channel_blocks(model):
    """State channel = all recurrent (delta) mixers; index channel = the
    attention mixer. Generic over 2- and 3-layer hybrids (the original version
    assumed blocks [state, attn] and, on 3-layer arms, probed/lesioned the
    second delta layer instead of attention — fixed here)."""
    state = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, DeltaMixer)]
    attn = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, AttnMixer)]
    assert state and len(attn) == 1, (state, attn)
    return state, attn[0]


def load_model(ck, arch, device):
    specs, _ = arch_specs(arch)
    m = SeqModel(VOCAB_E, VOCAB_E, 128, specs).to(device)
    m.load_state_dict(torch.load(ck, map_location=device))
    return m.eval()


@torch.no_grad()
def collect(model, device, seed, n_batches=40, bs=256, T=40, N=64):
    """Mixer outputs at (i) unplanted instruction positions -> r_k labels,
    (ii) QUERY position -> answer labels. State features come from the LAST
    delta mixer (the recurrent front's output); index features from the
    attention mixer."""
    state_blocks, attn_block = channel_blocks(model)
    probe_blocks = {'state': state_blocks[-1], 'attn': attn_block}
    feats = {k: {'r': [], 'v': []} for k in probe_blocks}
    lab_r, lab_v = [], []
    grabbed = {}
    hooks = [blk.mixer.register_forward_hook(
        (lambda i: (lambda m, inp, out: grabbed.__setitem__(i, out)))(i))
        for i, blk in enumerate(model.blocks)]
    rng = np.random.default_rng(424242 + seed)
    for _ in range(n_batches):
        toks, ans, aux = gen_comp_batch(bs, N, T, L, rng, device,
                                        return_states=True)
        model(toks)
        # instruction positions with ADDRESS targets (r_k unplanted)
        ipos = torch.arange(L - 1 - T, L - 1, device=device)
        is_addr = (aux >= E_P0) & (aux < E_P0 + 120)
        for name, ch in probe_blocks.items():
            g = grabbed[ch]
            sel = g[:, ipos, :][is_addr]                    # (K, D)
            feats[name]['r'].append(sel.float().cpu())
            feats[name]['v'].append(g[:, -1, :].float().cpu())
        lab_r.append((aux[is_addr] - E_P0).cpu())
        lab_v.append((ans - E_V0).cpu())
    for h in hooks:
        h.remove()
    out = {}
    for name in probe_blocks:
        out[name] = {k: torch.cat(v) for k, v in feats[name].items()}
    return out, torch.cat(lab_r), torch.cat(lab_v)


def probe(X, y, n_cls, device, epochs=60):
    """Multinomial logistic probe; 80/20 split; returns test accuracy."""
    n = X.shape[0]
    idx = torch.randperm(n)
    tr, te = idx[:int(0.8 * n)], idx[int(0.8 * n):]
    W = torch.zeros(X.shape[1], n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=3e-3)
    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (X[te].to(device) @ W + b).argmax(-1)
        return (pred == y[te].to(device)).float().mean().item()


@torch.no_grad()
def eval_cells(model, device, seed, cells=((256, 16), (40, 64)), n=1024, bs=256):
    out = {}
    for T, N in cells:
        rng = np.random.default_rng(888000 + seed * 1000 + T * 10 + N)
        acc = 0
        for _ in range(n // bs):
            toks, ans = gen_comp_batch(bs, N, T, L, rng, device)
            acc += (model(toks)[:, -1].argmax(-1) == ans).sum().item()
        out[f"T{T}_N{N}"] = acc / n
    return out


def lesion(model, which):
    """Zero a channel's mixer output(s); returns a restore handle.
    which='state' bypasses ALL recurrent mixers; which='attn' the attention."""
    state_blocks, attn_block = channel_blocks(model)
    idxs = state_blocks if which == 'state' else [attn_block]
    origs = []
    for i in idxs:
        blk = model.blocks[i]
        origs.append((blk.mixer, blk.mixer.forward))
        d = blk.ln1.normalized_shape[0]
        blk.mixer.forward = (lambda dd: (lambda x, *a, **k: torch.zeros(
            x.shape[0], x.shape[1], dd, device=x.device, dtype=x.dtype)))(d)

    def restore():
        for m, f in origs:
            m.forward = f
    return restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--arch', default='e_hybrid')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda'
    rows = []
    for ck in sorted(sum([glob.glob(c) for c in args.ckpts], [])):
        seed = int(ck.split('_s')[-1].split('_')[0])
        model = load_model(ck, args.arch, device)
        feats, yr, yv = collect(model, device, seed)
        pr = {}
        for name in ('state', 'attn'):
            pr[f'{name}_to_referent'] = round(
                probe(feats[name]['r'], yr, 120, device), 4)
            pr[f'{name}_to_value'] = round(
                probe(feats[name]['v'], yv, 256, device), 4)
        base = eval_cells(model, device, seed)
        les = {}
        for name in ('state', 'attn'):
            restore = lesion(model, name)
            les[name] = eval_cells(model, device, seed)
            restore()
        row = dict(ckpt=os.path.basename(ck), seed=seed, probes=pr,
                   base=base, lesion=les)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # frozen verdicts (amendment 10), on seed means
    def m(f):
        return float(np.mean([f(r) for r in rows]))
    v = {}
    v['a_state_decodes_referent'] = m(lambda r: r['probes']['state_to_referent'])
    v['a_attn_decodes_referent'] = m(lambda r: r['probes']['attn_to_referent'])
    v['a_state_decodes_value'] = m(lambda r: r['probes']['state_to_value'])
    v['a_attn_decodes_value'] = m(lambda r: r['probes']['attn_to_value'])
    v['a_pass'] = bool(
        v['a_state_decodes_referent'] - v['a_attn_decodes_referent'] >= 0.20
        and v['a_attn_decodes_value'] - v['a_state_decodes_value'] >= 0.20
        and v['a_state_decodes_referent'] >= 0.5
        and v['a_attn_decodes_value'] >= 0.5)
    dsT = m(lambda r: r['base']['T256_N16'] - r['lesion']['state']['T256_N16'])
    dsN = m(lambda r: r['base']['T40_N64'] - r['lesion']['state']['T40_N64'])
    daT = m(lambda r: r['base']['T256_N16'] - r['lesion']['attn']['T256_N16'])
    daN = m(lambda r: r['base']['T40_N64'] - r['lesion']['attn']['T40_N64'])
    v.update(b_state_lesion_drop_T=round(dsT, 4), b_state_lesion_drop_N=round(dsN, 4),
             b_attn_lesion_drop_T=round(daT, 4), b_attn_lesion_drop_N=round(daN, 4))
    v['b_pass'] = bool(dsT >= 0.5 and daN >= 0.5 and dsT > dsN and daN > daT)
    v['complementarity_reestablished'] = bool(v['a_pass'] and v['b_pass'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(rows=rows, verdicts=v), open(args.out, 'w'), indent=1)
    print(json.dumps(v, indent=1))


if __name__ == '__main__':
    main()
