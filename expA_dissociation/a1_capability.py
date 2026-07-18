"""Exp A1: Newton's-apple ICL retrieval on Pile-matched families.

All models (Pythia / Mamba / RWKV-4-pile) share the GPT-NeoX-20B tokenizer, so
every model sees the *identical* token stream: same data, same scale range, same
tokenizer — the only difference is access structure.

Prompt = [filler_pre | N facts | filler_post | query], exactly --length tokens.
Fact:  "Note: the secret code of {adj} {noun} is {value}.\n"
Query: "Question: What is the secret code of {adj} {noun}?\nAnswer: The secret
        code of {adj} {noun} is"  -> next token scored against " {value}".

Metrics per (N, depth) cell: exact-match top-1 accuracy and mean log-prob of the
value token. depth = fractional position of the fact block in the context.

Usage: python a1_capability.py --model state-spaces/mamba-370m-hf --length 896 \
           --out results/expA/a1_mamba-370m_L896.json
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch

ADJS = ("crimson azure golden silver ancient silent frozen hidden iron copper "
        "emerald violet scarlet obsidian amber ivory cobalt jade onyx pearl "
        "rustic solar lunar polar coastal alpine desert arctic tropical misty").split()
NOUNS = ("falcon tiger salmon walrus badger condor viper heron bison lynx "
         "otter raven moose cobra crane panda gecko jackal koala lemur "
         "marmot ocelot puffin quail rhino sable toucan urchin wombat yak").split()
VALUE_POOL = ("apple river stone cloud grass flame sugar pencil window garden "
              "bridge candle mirror basket ladder bottle circle planet forest "
              "silver copper marble velvet cotton butter pepper winter spring "
              "summer autumn morning evening thunder shadow crystal meadow "
              "harbor island valley canyon desert prairie tundra lagoon "
              "engine hammer needle ribbon saddle teapot violin wagon anchor").split()

N_GRID = [4, 8, 16, 32, 64]
DEPTHS = [0.1, 0.5, 0.9]


def build_instances(tok, length, n_per_cell, seed):
    """Pre-build token-level prompts; identical across models (shared tokenizer)."""
    rng = random.Random(seed)
    # filler pool: pile-10k documents, tokenized once
    from datasets import load_dataset
    docs = load_dataset("NeelNanda/pile-10k", split="train")
    filler_ids = []
    for d in docs.select(range(400)):
        ids = tok(d["text"], add_special_tokens=False).input_ids
        filler_ids.extend(ids)
        if len(filler_ids) > 3_000_000:
            break

    # single-token values (with leading space)
    values = [v for v in VALUE_POOL
              if len(tok(" " + v, add_special_tokens=False).input_ids) == 1]
    assert len(values) >= 40, f"only {len(values)} single-token values"

    cells = {}
    for N in N_GRID:
        # ~12 tokens per fact; require >=15% of context left as filler
        if N * 13 > 0.85 * length:
            print(f"skip N={N} (does not fit length {length})")
            continue
        for depth in DEPTHS:
            insts = []
            for i in range(n_per_cell):
                names = rng.sample([f"{a} {n}" for a in ADJS for n in NOUNS], N)
                vals = [rng.choice(values) for _ in range(N)]
                qi = rng.randrange(N)
                facts = "".join(f"Note: the secret code of {nm} is {vv}.\n"
                                for nm, vv in zip(names, vals))
                query = (f"\nQuestion: What is the secret code of {names[qi]}?\n"
                         f"Answer: The secret code of {names[qi]} is")
                f_ids = tok(facts, add_special_tokens=False).input_ids
                q_ids = tok(query, add_special_tokens=False).input_ids
                ans_id = tok(" " + vals[qi], add_special_tokens=False).input_ids[0]
                n_fill = length - len(f_ids) - len(q_ids)
                if n_fill < 0:
                    raise ValueError(f"N={N} facts do not fit in length {length}")
                pre = int(n_fill * depth)
                start = rng.randrange(0, len(filler_ids) - n_fill - 1)
                fill = filler_ids[start:start + n_fill]
                ids = fill[:pre] + f_ids + fill[pre:] + q_ids
                assert len(ids) == length
                insts.append((ids, ans_id))
            cells[f"N{N}_d{depth}"] = insts
    return cells


@torch.no_grad()
def score(model, cells, device, bs):
    out = {}
    for cell, insts in cells.items():
        acc, lp, n = 0, 0.0, len(insts)
        for i in range(0, n, bs):
            batch = insts[i:i + bs]
            ids = torch.tensor([b[0] for b in batch], device=device)
            ans = torch.tensor([b[1] for b in batch], device=device)
            logits = model(input_ids=ids).logits[:, -1].float()
            logp = torch.log_softmax(logits, -1)
            acc += (logits.argmax(-1) == ans).sum().item()
            lp += logp[torch.arange(len(batch)), ans].sum().item()
        out[cell] = dict(acc=acc / n, logp=lp / n)
        print(f"  {cell}: acc={acc/n:.3f} logp={lp/n:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--length', type=int, default=896)
    ap.add_argument('--n_per_cell', type=int, default=100)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    cells = build_instances(tok, args.length, args.n_per_cell, args.seed)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to('cuda').eval()
    res = score(model, cells, 'cuda', args.bs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(model=args.model, length=args.length,
                   n_per_cell=args.n_per_cell, seed=args.seed, cells=res,
                   wall_seconds=round(time.time() - t0)),
              open(args.out, 'w'), indent=1)
    print("DONE", args.out, flush=True)


if __name__ == '__main__':
    main()
