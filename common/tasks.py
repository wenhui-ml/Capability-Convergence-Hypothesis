"""Task generators for CCH experiments.

Exp C: S5 word problem (state tracking).
Exp B: executable NA(N, b, B; kappa) — Newton's-apple MQAR with semantic-overlap
       distractors and a compressibility axis.
"""
import itertools
import numpy as np
import torch

# ---------------- S5 (Exp C) ----------------

PERMS = list(itertools.permutations(range(5)))          # 120 elements, index = token id
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
# composition: (a*b)(i) = a[b[i]]  (apply b first, then a)
COMPOSE = np.zeros((120, 120), dtype=np.int64)
for ia, a in enumerate(PERMS):
    for ib, b in enumerate(PERMS):
        COMPOSE[ia, ib] = PERM_IDX[tuple(a[b[i]] for i in range(5))]

TRANSPOSITIONS = []
for i in range(5):
    for j in range(i + 1, 5):
        p = list(range(5))
        p[i], p[j] = p[j], p[i]
        TRANSPOSITIONS.append(PERM_IDX[tuple(p)])       # 10 generators
IDENTITY = PERM_IDX[tuple(range(5))]


def gen_s5_batch(batch, T, stream, rng, device):
    """stream in {'swap','general'}. tokens (B,T) = element ids; labels (B,T) =
    cumulative product ids (state after consuming token t)."""
    if stream == 'swap':
        toks = rng.choice(TRANSPOSITIONS, size=(batch, T))
    else:
        toks = rng.integers(0, 120, size=(batch, T))
    labels = np.zeros_like(toks)
    state = np.full(batch, IDENTITY, dtype=np.int64)
    for t in range(T):
        state = COMPOSE[state, toks[:, t]]
        labels[:, t] = state
    return (torch.from_numpy(toks).to(device),
            torch.from_numpy(labels).to(device))


# ---------------- Newton-apple MQAR (Exp B) ----------------
# Vocab: 0 PAD, 1 SEP, 2 QUERY, cluster tokens [3, 3+NC), id tokens [3+NC, 3+NC+NI),
# value tokens [3+NC+NI, 3+NC+NI+NV).
NC, NI, NV = 64, 8, 256          # 64 clusters x 8 ids = 512 composite keys; b = 8 bits
PAD, SEP, QUERY = 0, 1, 2
C0, I0, V0 = 3, 3 + NC, 3 + NC + NI
VOCAB_B = 3 + NC + NI + NV       # 331
PLANT_CLUSTERS = 48              # bindings drawn from clusters [0,48); low-kappa noise from [48,64)


def gen_na_batch(batch, N, L, kappa, rng, device, value_block=1, bindings_late=False,
                 collide=0):
    """Sequence: [c u v]*N  SEP  [c u]*M(noise)  QUERY c_q u_q  -> answer v_q.
    Returns tokens (B, L) and answers (B,). Answer position is the last token.
    kappa: 'high' (noise keys from planted clusters) or 'low' (from disjoint clusters).
    value_block: consecutive planted keys share one value in blocks of this size
                 (compressibility axis; effective value entropy ~ b/value_block).
    bindings_late: noise first, bindings adjacent to the query (horizon-wall probe).
    collide: Assumption-sep negative control (amendment 8). Each planted key is
             bound collide+1 times to DISTINCT values, triples shuffled through the
             binding block; the labeled 'true' value is a uniformly random one of
             them and nothing in the stream marks it. Code distance between target
             and colliding distractors is exactly zero, so the Bayes floor is
             err = collide/(collide+1), nll = log2(collide+1) bits, independent
             of N and of architecture.
    """
    R = collide + 1
    M = (L - 3 * N * R - 4) // 2                 # noise pairs
    assert M >= 0, f"N={N} (x{R} bindings) too large for L={L}"
    toks = np.full((batch, L), PAD, dtype=np.int64)

    # planted composite keys: (cluster, id), unique per instance
    key_flat = np.stack([rng.choice(PLANT_CLUSTERS * NI, size=N, replace=False)
                         for _ in range(batch)])          # (B,N) in [0, 384)
    kc, ki = key_flat // NI, key_flat % NI
    if collide == 0:
        # values with compressibility blocks
        nblocks = (N + value_block - 1) // value_block
        vals_b = rng.integers(0, NV, size=(batch, nblocks))
        vals = np.repeat(vals_b, value_block, axis=1)[:, :N]  # (B,N)

        bind = np.empty((batch, 3 * N), dtype=np.int64)
        bind[:, 0::3], bind[:, 1::3], bind[:, 2::3] = C0 + kc, I0 + ki, V0 + vals
    else:
        assert value_block == 1, "collide and value_block are separate axes"
        # R distinct values per key (uniform without replacement), true one hidden
        vals_all = np.argpartition(rng.random((batch, N, NV)), R, axis=2)[:, :, :R]
        true_r = rng.integers(0, R, size=(batch, N))
        vals = np.take_along_axis(vals_all, true_r[..., None], axis=2)[..., 0]  # (B,N)
        # N*R triples, shuffled per instance so order carries no signal
        kc_rep = np.repeat(kc, R, axis=1)
        ki_rep = np.repeat(ki, R, axis=1)
        v_rep = vals_all.reshape(batch, N * R)
        perm = rng.permuted(np.tile(np.arange(N * R), (batch, 1)), axis=1)
        bind = np.empty((batch, 3 * N * R), dtype=np.int64)
        bind[:, 0::3] = C0 + np.take_along_axis(kc_rep, perm, axis=1)
        bind[:, 1::3] = I0 + np.take_along_axis(ki_rep, perm, axis=1)
        bind[:, 2::3] = V0 + np.take_along_axis(v_rep, perm, axis=1)

    # noise pairs (unbound keys)
    if M > 0:
        if kappa == 'high':
            # clusters sampled from each instance's planted clusters
            idx = rng.integers(0, N, size=(batch, M))
            nc_tok = np.take_along_axis(kc, idx, axis=1)
            ni_tok = rng.integers(0, NI, size=(batch, M))
            # avoid exact collision with the planted key of that cluster instance
            coll = ni_tok == np.take_along_axis(ki, idx, axis=1)
            ni_tok[coll] = (ni_tok[coll] + 1) % NI
        else:
            nc_tok = rng.integers(PLANT_CLUSTERS, NC, size=(batch, M))
            ni_tok = rng.integers(0, NI, size=(batch, M))
        noise = np.empty((batch, 2 * M), dtype=np.int64)
        noise[:, 0::2], noise[:, 1::2] = C0 + nc_tok, I0 + ni_tok
    else:
        noise = np.empty((batch, 0), dtype=np.int64)

    # query one planted key uniformly
    qi = rng.integers(0, N, size=batch)
    qc = C0 + kc[np.arange(batch), qi]
    qu = I0 + ki[np.arange(batch), qi]
    answers = V0 + vals[np.arange(batch), qi]

    parts = ([noise, np.full((batch, 1), SEP), bind] if bindings_late
             else [bind, np.full((batch, 1), SEP), noise])
    body = np.concatenate(parts + [np.full((batch, 1), QUERY),
                                   qc[:, None], qu[:, None]], axis=1)
    toks[:, -body.shape[1]:] = body                      # left-pad with PAD
    return (torch.from_numpy(toks).to(device),
            torch.from_numpy(answers).to(device))


# ---------------- Composite witness: NA composed with S5 (Exp E, amendment 9) --
# Vocab: 0 PAD, 1 SEP, 2 QUERY, perm-address tokens [3, 123), swap-instruction
# tokens [123, 133) (the 10 transpositions), value tokens [133, 133+NV_E).
NV_E = 256
E_P0, E_I0, E_V0 = 3, 123, 133
VOCAB_E = E_V0 + NV_E                               # 389
_TRANS = np.array(TRANSPOSITIONS)


def gen_comp_batch(batch, N, T, L, rng, device, return_states=False):
    """[p v]*N  SEP  t_1..t_T  QUERY  -> value bound to r_T = t_1 o ... o t_T.

    Bindings are written FIRST (distance to query > T, outside any w=32 window);
    a swap-instruction stream then drives the referent; the queried address is
    the COMPOSED final referent, guaranteed planted among the N addresses.
    Solving requires state tracking (compose r_T, length-extrapolating) AND
    exact retrieval (N x 8-bit bindings across the instruction span) in one
    prediction — the witness conjunction of Prop. nonpreserve. Answer at the
    last (QUERY) position.

    return_states: also return (B,T) per-instruction-position dense targets for
    amendment 9 (the composite analogue of Exp C's per-position labels, applied
    uniformly to every arm): at instruction position k the target is the VALUE
    token bound to the current referent r_k when r_k is planted, else the
    ADDRESS token of r_k. Tracking and retrieval-coupling are both densely
    supervised; the final answer equals the running answer at the last step.
    """
    instr = rng.integers(0, 10, size=(batch, T))
    r = np.full(batch, IDENTITY, dtype=np.int64)
    states = np.empty((batch, T), dtype=np.int64) if return_states else None
    for t in range(T):
        r = COMPOSE[r, _TRANS[instr[:, t]]]
        if return_states:
            states[:, t] = r
    # N distinct addresses, with r_T planted in a random slot when absent
    P = np.stack([rng.choice(120, size=N, replace=False) for _ in range(batch)])
    has = (P == r[:, None]).any(1)
    slot = rng.integers(0, N, size=batch)
    rows = np.where(~has)[0]
    P[rows, slot[rows]] = r[rows]
    vals = rng.integers(0, NV_E, size=(batch, N))
    qi = (P == r[:, None]).argmax(1)
    answers = E_V0 + vals[np.arange(batch), qi]

    bind = np.empty((batch, 2 * N), dtype=np.int64)
    bind[:, 0::2], bind[:, 1::2] = E_P0 + P, E_V0 + vals
    body = np.concatenate([bind, np.full((batch, 1), SEP), E_I0 + instr,
                           np.full((batch, 1), QUERY)], axis=1)
    assert body.shape[1] <= L, f"N={N}, T={T} does not fit L={L}"
    toks = np.full((batch, L), PAD, dtype=np.int64)
    toks[:, -body.shape[1]:] = body
    if return_states:
        # instruction tokens sit at positions [L-1-T, L-2] (QUERY at L-1);
        # per-position target: bound value if current referent planted, else
        # its address token ("running answer" supervision)
        vmap = np.full((batch, 120), -1, dtype=np.int64)
        np.put_along_axis(vmap, P, vals, axis=1)
        bound = np.take_along_axis(vmap, states, axis=1)         # (B,T)
        aux = np.where(bound >= 0, E_V0 + bound, E_P0 + states)
        return (torch.from_numpy(toks).to(device),
                torch.from_numpy(answers).to(device),
                torch.from_numpy(aux).to(device))
    return (torch.from_numpy(toks).to(device),
            torch.from_numpy(answers).to(device))
