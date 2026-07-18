"""Shared model zoo for CCH experiments B/C/D.

One backbone, swappable sequence mixers. All mixers are causal. Pure PyTorch
(sequential scan for recurrent mixers) so that beta range / n_h / state size are
fully controlled — no external kernel dependencies.

Mixer types:
  delta      — (Gated) DeltaNet / DeltaProduct: n_h rank-1 Householder-style delta
               updates per token, beta in (0, beta_scale). beta_scale=1 -> eigenvalues
               of the state transition in (0,1) (no reflection); beta_scale=2 ->
               eigenvalues in (-1,1) (reflection unlocked).
  diag       — selective gated diagonal SSM (Mamba-style abstraction).
  attn       — full softmax attention with RoPE.
  swa        — sliding-window softmax attention with RoPE (window w).
  lstm       — nn.LSTM control.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class DeltaMixer(nn.Module):
    """DeltaNet (n_h=1) / DeltaProduct (n_h>1) with controllable beta range.

    State S: (B, H, dk, dv) per head. Per token, n_h sequential updates:
        S <- S + beta_j * k_j (v_j - S^T k_j)^T   == (I - beta k k^T) S + beta k v^T
    Read: y_t = S_t^T q_t.
    """

    def __init__(self, d_model, dk, dv, n_heads=1, n_h=1, beta_scale=2.0, conv=4):
        super().__init__()
        self.h, self.dk, self.dv, self.n_h = n_heads, dk, dv, n_h
        self.beta_scale = beta_scale
        # standard short causal depthwise conv (as in Mamba/GLA/DeltaNet production
        # parameterizations); local window `conv` << any binding-query distance
        self.conv = nn.Conv1d(d_model, d_model, conv, padding=conv - 1,
                              groups=d_model) if conv else None
        self.Wq = nn.Linear(d_model, n_heads * dk, bias=False)
        self.Wk = nn.Linear(d_model, n_heads * n_h * dk, bias=False)
        self.Wv = nn.Linear(d_model, n_heads * n_h * dv, bias=False)
        self.Wb = nn.Linear(d_model, n_heads * n_h, bias=True)
        self.Wg = nn.Linear(d_model, n_heads * dv, bias=False)
        self.Wo = nn.Linear(n_heads * dv, d_model, bias=False)

    def state_scalars(self):
        return self.h * self.dk * self.dv

    def forward(self, x, sequential=False):
        B, T, _ = x.shape
        H, dk, dv, nh = self.h, self.dk, self.dv, self.n_h
        if self.conv is not None:
            x = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        q = l2norm(self.Wq(x).view(B, T, H, dk))
        k = l2norm(self.Wk(x).view(B, T, H, nh, dk))
        v = self.Wv(x).view(B, T, H, nh, dv)
        beta = self.beta_scale * torch.sigmoid(self.Wb(x)).view(B, T, H, nh)
        if sequential:
            y = self._scan_sequential(q, k, v, beta)
        else:
            y = self._scan_chunked(q, k, v, beta)
        y = y.reshape(B, T, H * dv)
        y = y * F.silu(self.Wg(x))
        return self.Wo(y)

    def _scan_sequential(self, q, k, v, beta):
        """Reference implementation: explicit rank-1 delta updates."""
        B, T, H, _ = q.shape
        S = q.new_zeros(B, H, self.dk, self.dv)
        ys = []
        for t in range(T):
            for j in range(self.n_h):
                kj = k[:, t, :, j]                        # (B,H,dk)
                vj = v[:, t, :, j]                        # (B,H,dv)
                bj = beta[:, t, :, j].unsqueeze(-1)       # (B,H,1)
                v_old = torch.einsum('bhkv,bhk->bhv', S, kj)
                S = S + torch.einsum('bhk,bhv->bhkv', kj, bj * (vj - v_old))
            ys.append(torch.einsum('bhkv,bhk->bhv', S, q[:, t]))
        return torch.stack(ys, dim=1)                     # (B,T,H,dv)

    def _scan_chunked(self, q, k, v, beta, C=64):
        """Exact chunked WY-representation scan (Yang et al., parallel DeltaNet).

        Flatten the T*n_h rank-1 updates into one stream; within a chunk of size C
        solve the unit-lower-triangular system
            (I + diag(beta) tril(K K^T, -1)) U = diag(beta) (V - K S0)
        so that S_end = S0 + K^T U, and reads at position i (after its own token's
        updates) are Y_i = q_i S0 + [tril(Q K^T)]_i U. Identical to the sequential
        scan up to float associativity.
        """
        B, T, H, dk = q.shape
        nh, dv = self.n_h, self.dv
        Fl = T * nh
        kf = k.permute(0, 2, 1, 3, 4).reshape(B, H, Fl, dk)
        vf = v.permute(0, 2, 1, 3, 4).reshape(B, H, Fl, dv)
        bf = beta.permute(0, 2, 1, 3).reshape(B, H, Fl)
        # queries live at flattened positions t*nh + nh-1 (read after own update)
        qf = q.new_zeros(B, H, Fl, dk)
        qf[:, :, nh - 1::nh] = q.permute(0, 2, 1, 3)
        pad = (-Fl) % C
        if pad:
            kf = F.pad(kf, (0, 0, 0, pad))
            vf = F.pad(vf, (0, 0, 0, pad))
            bf = F.pad(bf, (0, pad))                      # beta=0 -> no-op update
            qf = F.pad(qf, (0, 0, 0, pad))
        S = q.new_zeros(B, H, dk, dv)
        eye = torch.eye(C, device=q.device, dtype=q.dtype)
        tril_mask = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
        ys = []
        for c in range(0, kf.shape[2], C):
            Kc = kf[:, :, c:c + C]
            Vc = vf[:, :, c:c + C]
            bc = bf[:, :, c:c + C]
            Qc = qf[:, :, c:c + C]
            KK = Kc @ Kc.transpose(-1, -2)
            M = eye + bc.unsqueeze(-1) * KK.tril(-1)
            rhs = bc.unsqueeze(-1) * (Vc - Kc @ S)
            U = torch.linalg.solve_triangular(M, rhs, upper=False)
            QK = (Qc @ Kc.transpose(-1, -2)).masked_fill(~tril_mask, 0)
            ys.append(Qc @ S + QK @ U)
            S = S + Kc.transpose(-1, -2) @ U
        Y = torch.cat(ys, dim=2)[:, :, :Fl]               # (B,H,Fl,dv)
        return Y[:, :, nh - 1::nh].permute(0, 2, 1, 3)    # (B,T,H,dv)


class DiagSSMMixer(nn.Module):
    """Selective gated diagonal recurrence: h_t = a_t*h_{t-1} + (1-a_t)*u_t, a_t in (0,1)."""

    def __init__(self, d_model, d_h, conv=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, conv, padding=conv - 1,
                              groups=d_model) if conv else None
        self.Wa = nn.Linear(d_model, d_h, bias=True)
        self.Wu = nn.Linear(d_model, d_h, bias=False)
        self.Wg = nn.Linear(d_model, d_h, bias=False)
        self.Wo = nn.Linear(d_h, d_model, bias=False)
        nn.init.constant_(self.Wa.bias, 2.0)  # start with slow decay

    def forward(self, x):
        B, T, _ = x.shape
        if self.conv is not None:
            x = F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))
        a = torch.sigmoid(self.Wa(x))
        u = self.Wu(x)
        h = x.new_zeros(B, a.shape[-1])
        ys = []
        for t in range(T):
            h = a[:, t] * h + (1 - a[:, t]) * u[:, t]
            ys.append(h)
        y = torch.stack(ys, dim=1) * F.silu(self.Wg(x))
        return self.Wo(y)


def rope(qk, base=10000.0):
    # qk: (B, H, T, d) -> rotate half
    B, H, T, d = qk.shape
    half = d // 2
    freqs = torch.arange(half, device=qk.device, dtype=torch.float32)
    inv = base ** (-freqs / half)
    t = torch.arange(T, device=qk.device, dtype=torch.float32)
    ang = torch.outer(t, inv)                      # (T, half)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = qk[..., :half], qk[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class AttnMixer(nn.Module):
    def __init__(self, d_model, n_heads=4, window=None):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.window = window
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_attn_repr=False):
        B, T, D = x.shape
        qkv = self.Wqkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = rope(qkv[0]), rope(qkv[1]), qkv[2]
        if self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            i = torch.arange(T, device=x.device)
            mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.Wo(y)


class LSTMMixer(nn.Module):
    def __init__(self, d_model, d_h):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_h, batch_first=True)
        self.Wo = nn.Linear(d_h, d_model, bias=False)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.Wo(y)


def make_mixer(spec, d_model):
    kind = spec['kind']
    if kind == 'delta':
        return DeltaMixer(d_model, spec['dk'], spec['dv'], spec.get('heads', 1),
                          spec.get('n_h', 1), spec.get('beta_scale', 2.0))
    if kind == 'diag':
        return DiagSSMMixer(d_model, spec.get('d_h', 2 * d_model))
    if kind == 'attn':
        return AttnMixer(d_model, spec.get('heads', 4), window=None)
    if kind == 'swa':
        return AttnMixer(d_model, spec.get('heads', 4), window=spec['window'])
    if kind == 'lstm':
        return LSTMMixer(d_model, spec.get('d_h', 2 * d_model))
    raise ValueError(kind)


class Block(nn.Module):
    def __init__(self, d_model, mixer_spec, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = make_mixer(mixer_spec, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SeqModel(nn.Module):
    """Embedding -> blocks -> LN -> linear head (vocab_out classes per position)."""

    def __init__(self, vocab_in, vocab_out, d_model, mixer_specs, mlp_ratio=2,
                 tie_embeddings=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d_model)
        self.blocks = nn.ModuleList([Block(d_model, s, mlp_ratio) for s in mixer_specs])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_out, bias=False)
        if tie_embeddings:
            # standard weight tying; in Exp E it aligns the state channel's
            # decoded-address space with the input-embedding key space that the
            # retrieval attention matches against (applied uniformly to all arms)
            assert vocab_in == vocab_out
            self.head.weight = self.emb.weight

    def forward(self, tokens, return_layer_outputs=False):
        x = self.emb(tokens)
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        logits = self.head(self.ln_f(x))
        if return_layer_outputs:
            return logits, outs
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------- architecture registry (Exp B & C) ----------------

def arch_specs(name, d_model=128):
    """Return (mixer_specs, description). State sizes count TOTAL recurrent scalars."""
    A = {
        # ---- Exp C (S5): 2 layers, matched width ----
        'c_delta_b1':   ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=1.0)] * 2,
                         'DeltaNet beta(0,1) n_h=1'),
        'c_delta_b2':   ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0)] * 2,
                         'DeltaNet beta(0,2) n_h=1'),
        'c_dp2':        ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=2, beta_scale=2.0)] * 2,
                         'DeltaProduct n_h=2 beta(0,2)'),
        'c_dp4':        ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=4, beta_scale=2.0)] * 2,
                         'DeltaProduct n_h=4 beta(0,2)'),
        'c_diag':       ([dict(kind='diag', d_h=256)] * 2, 'diagonal SSM'),
        'c_attn':       ([dict(kind='attn', heads=4)] * 2, 'Transformer (RoPE)'),
        'c_attn8':      ([dict(kind='attn', heads=4)] * 8,
                         'Transformer 8L (RoPE, depth-advantaged control)'),
        'c_lstm':       ([dict(kind='lstm', d_h=256)] * 2, 'LSTM'),
        # ---- Exp B (floor): 2 mixer layers ----
        # pure state, total state m = 2 * heads * dk * dv scalars
        'b_state64':    ([dict(kind='delta', dk=8, dv=4, heads=1, n_h=1, beta_scale=2.0)] * 2,
                         'pure state m=64'),
        'b_state256':   ([dict(kind='delta', dk=16, dv=8, heads=1, n_h=1, beta_scale=2.0)] * 2,
                         'pure state m=256'),
        'b_state1024':  ([dict(kind='delta', dk=32, dv=16, heads=1, n_h=1, beta_scale=2.0)] * 2,
                         'pure state m=1024'),
        'b_swa':        ([dict(kind='swa', heads=4, window=32)] * 2, 'sliding window w=32'),
        'b_hybrid':     ([dict(kind='delta', dk=8, dv=8, heads=1, n_h=1, beta_scale=2.0),
                          dict(kind='attn', heads=4)],
                         'hybrid: state m=64 + 1 global attn'),
        'b_hybrid_local': ([dict(kind='delta', dk=8, dv=8, heads=1, n_h=1, beta_scale=2.0),
                            dict(kind='swa', heads=4, window=32)],
                           'state m=64 + local window only (NOT access-complete)'),
        'b_hybrid_b1':  ([dict(kind='delta', dk=8, dv=8, heads=1, n_h=1, beta_scale=1.0),
                          dict(kind='attn', heads=4)],
                         'hybrid: state m=64 beta(0,1) + 1 global attn (Exp D failing arm)'),
        'b_attn':       ([dict(kind='attn', heads=4)] * 2, 'full attention'),
        # ---- Exp E (composite witness, amendment 9): 2 mixer layers ----
        # state channel sized to track S5 but stay under the load at N=64
        # (m = 2*2*16*16 = 1024 scalars pure / 512 in the hybrid's single layer)
        'e_state':      ([dict(kind='delta', dk=16, dv=16, heads=2, n_h=1, beta_scale=2.0)] * 2,
                         'pure state beta(0,2), m=1024 (composite witness)'),
        'e_hybrid':     ([dict(kind='delta', dk=16, dv=16, heads=2, n_h=1, beta_scale=2.0),
                          dict(kind='attn', heads=4)],
                         'hybrid: tracking state m=512 + 1 global attn'),
        'e_hybrid_b1':  ([dict(kind='delta', dk=16, dv=16, heads=2, n_h=1, beta_scale=1.0),
                          dict(kind='attn', heads=4)],
                         'hybrid with beta(0,1) state (no reflection): tracking arm broken'),
        'e_hybrid_local': ([dict(kind='delta', dk=16, dv=16, heads=2, n_h=1, beta_scale=2.0),
                            dict(kind='swa', heads=4, window=32)],
                           'tracking state + local window only (NOT access-complete)'),
        'e_hybrid_big': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                          dict(kind='attn', heads=4)],
                         'hybrid: Exp-C-size tracking state (m=2048) + 1 global attn'),
        'e_hybrid3': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                       dict(kind='attn', heads=4), dict(kind='attn', heads=4)],
                      'hybrid: tracking state + 2 global attn (selection depth)'),
        'e_state3': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0)] * 3,
                     'pure state beta(0,2), 3 layers m=6144'),
        'e_hybrid_d2a': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                          dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                          dict(kind='attn', heads=4)],
                         'hybrid: 2-layer recurrent front (tracking) + 1 global attn (retrieval)'),
        'e_state3_small': ([dict(kind='delta', dk=16, dv=16, heads=2, n_h=1, beta_scale=2.0)] * 3,
                           'pure state beta(0,2), 3 layers m=1536 (tight capacity)'),
        'e_hybrid_b1_d2a': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=1.0),
                             dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=1.0),
                             dict(kind='attn', heads=4)],
                            'd2a hybrid with beta(0,1) recurrent front: tracking arm broken'),
        'e_hybrid_local_d2a': ([dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                                dict(kind='delta', dk=32, dv=32, heads=2, n_h=1, beta_scale=2.0),
                                dict(kind='swa', heads=4, window=32)],
                               'd2a with window-32 index only (NOT access-complete)'),
    }
    return A[name]
