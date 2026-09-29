"""GPU objectives, Pareto filtering, compromise selection, ARI.

Objectives (all minimised), cover Psi = {P_1..P_k}, path c_r = R(L(P_r)), u_r = (L^U-L^L)/(L^M+eps):
  f1 = k
  f2 = max_r c_r                                   worst-path fuzzy cost
  f3 = 0.7 * mean_r u_r + 0.3 * Q0.90(u_1..u_k)    uncertainty (mean + tail risk)
  f4 = -(1/k) sum_r s_r,  s_r = (prod_e mu_E(e))^(1/|P_r|)   negative mean confidence
Path statistics depend on BOTH memberships and fuzzy lengths, so both can be perturbed.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch

# All sparse tensors below are built from validated indices; enable the invariant checks explicitly
# (instead of relying on the implicit default) and silence the framework notice for clean logs.
warnings.filterwarnings("ignore", message=".*[Ss]parse invariant checks.*")
try:
    torch.sparse.check_sparse_tensor_invariants.enable()
except Exception:
    pass

from .tfn import rank_arrays, rank_tensors

F64 = torch.float64
F3_MEAN_W, F3_TAIL_W, F3_Q = 0.7, 0.3, 0.90


class PoolTensors:
    def __init__(self, pool, cfg, device):
        self.device, self.eps, self.rank_mode = device, cfg.eps, cfg.rank_mode
        self.cost_np = rank_arrays(pool.sumL, pool.sumM, pool.sumU, cfg.rank_mode)   # nominal c_r
        self.Ur_np = (pool.sumU - pool.sumL) / (pool.sumM + cfg.eps)                # nominal u_r
        pad = torch.as_tensor(pool.edge_pad, device=device)
        self.emask = pad >= 0
        self.epad = pad.clamp(min=0)
        self.nedges = torch.as_tensor(pool.n_edges, dtype=F64, device=device)
        self.P, self.maxlen = pad.shape
        rows = np.repeat(np.arange(self.P), pool.n_edges)
        cols = np.concatenate([np.asarray(e) for e in pool.edges])
        idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
        self.Inc = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], dtype=F64, device=device),
                                           (self.P, pool.single_id.shape[0])).coalesce()


def path_conf(pt, mu_e):
    """mu_e: (B,E) -> s: (P,B) geometric-mean edge membership of every pool path."""
    logm = torch.log(mu_e.clamp(min=1e-6, max=1.0))
    B = logm.shape[0]
    chunk = max(1, int(2e7 / (pt.P * pt.maxlen)))
    out = []
    for s in range(0, B, chunk):
        lm = logm[s:s + chunk][:, pt.epad]
        lm = torch.where(pt.emask[None], lm, torch.zeros((), dtype=F64, device=pt.device))
        out.append(torch.exp(lm.sum(-1) / pt.nedges[None]))
    return torch.cat(out, 0).T.contiguous()


def path_stats(pt, mu_e, Lb):
    """mu_e (B,E), Lb (B,E,3) -> s, c, u each (P,B).  Fuzzy addition of lengths via sparse matmul."""
    s = path_conf(pt, mu_e)
    B, E, _ = Lb.shape
    X = Lb.permute(1, 0, 2).reshape(E, B * 3)
    sums = torch.sparse.mm(pt.Inc, X).reshape(pt.P, B, 3)
    c = rank_tensors(sums, pt.rank_mode)
    u = (sums[..., 2] - sums[..., 0]) / (sums[..., 1] + pt.eps)
    return s, c, u


class Candidates:
    """The shared candidate set C with GPU-resident structure for fast (re-)evaluation."""

    def __init__(self, pool, covers, cfg, device):
        self.pool, self.covers, self.device = pool, covers, device
        self.pt = PoolTensors(pool, cfg, device)
        C = len(covers)
        self.C = C
        lens_np = np.array([len(c) for c in covers])
        kmax = int(lens_np.max())
        pad = -np.ones((C, kmax), dtype=np.int64)
        for i, c in enumerate(covers):
            pad[i, :len(c)] = c
        pad_t = torch.as_tensor(pad, device=device)
        self.mask = pad_t >= 0
        self.padc = pad_t.clamp(min=0)
        self.kmax = kmax
        self.k = torch.as_tensor(lens_np, dtype=F64, device=device)
        pos = F3_Q * (self.k - 1)
        self.q_lo = pos.floor().long()
        self.q_hi = pos.ceil().long()
        self.q_fr = pos - pos.floor()
        cov_idx = torch.repeat_interleave(torch.arange(C, device=device), torch.as_tensor(lens_np, device=device))
        ids = torch.as_tensor(np.concatenate(covers), dtype=torch.long, device=device)
        self.M = torch.sparse_coo_tensor(torch.stack([cov_idx, ids]), (1.0 / self.k)[cov_idx],
                                         (C, self.pt.P)).coalesce()        # rows average over paths
        self.labels_cache = {}

    def objectives(self, s, c, u):
        """s,c,u: (P,B) -> F (B,C,4), meanU (B,C)."""
        B = s.shape[1]
        S = torch.sparse.mm(self.M, s)                                     # (C,B)
        mU = torch.sparse.mm(self.M, u)                                    # (C,B)
        f2 = torch.empty(B, self.C, dtype=F64, device=s.device)
        q90 = torch.empty(B, self.C, dtype=F64, device=s.device)
        chunk = max(1, int(1.5e7 / (self.C * self.kmax)))
        inf = float("inf")
        ar = torch.arange(self.C, device=s.device)
        for b0 in range(0, B, chunk):
            cb, ub = c.T[b0:b0 + chunk], u.T[b0:b0 + chunk]                # (b,P)
            f2[b0:b0 + chunk] = cb[:, self.padc].masked_fill(~self.mask[None], -inf).amax(-1)
            us, _ = torch.sort(ub[:, self.padc].masked_fill(~self.mask[None], inf), dim=-1)
            lo = us[:, ar, self.q_lo]
            hi = us[:, ar, self.q_hi]
            q90[b0:b0 + chunk] = lo * (1 - self.q_fr) + hi * self.q_fr
            del us
        f3 = F3_MEAN_W * mU.T + F3_TAIL_W * q90
        F = torch.stack([self.k.expand(B, -1), f2, f3, -S.T], dim=2)
        return F, mU.T


def pareto_front(F, chunk=2048):
    """Indices (into F) of one representative per non-dominated unique objective vector."""
    Fr = torch.round(F * 1e9) / 1e9
    uniq, inv = torch.unique(Fr, dim=0, return_inverse=True)
    Cn = F.shape[0]
    first = torch.full((uniq.shape[0],), Cn, dtype=torch.long, device=F.device)
    first.scatter_reduce_(0, inv, torch.arange(Cn, device=F.device), reduce="amin")
    dominated = torch.zeros(uniq.shape[0], dtype=torch.bool, device=F.device)
    for s in range(0, uniq.shape[0], chunk):
        Bm = uniq[s:s + chunk]
        le = (uniq[:, None, :] <= Bm[None, :, :]).all(-1)
        dominated[s:s + chunk] = le.sum(0) > 1
    return first[~dominated].cpu().numpy()


def compromise_select(Ff, lam=(0.25, 0.25, 0.25, 0.25), eps=1e-9):
    """Balanced solution on a Pareto set: normalise each objective by ideal/nadir OF THAT SET,
    f^_j = (f_j - ideal_j)/(nadir_j - ideal_j + eps), then minimise D = sqrt(sum_j lam_j f^_j^2)
    (lam = 1/4 each gives D = sqrt(1/4 sum f^_j^2)).  Returns (index into Ff, indices of all solutions tied with it)."""
    lam_t = torch.as_tensor(lam, dtype=F64, device=Ff.device)
    I, N = Ff.min(0).values, Ff.max(0).values
    fb = (Ff - I) / (N - I + eps)
    D = ((lam_t * fb ** 2).sum(1)).sqrt()
    i = int(torch.argmin(D).item())
    ties = torch.where(D <= D[i] + 1e-9)[0].cpu().numpy()          # equally good compromise solutions
    return i, ties


def nondominated_and_distance(Fb, i0, scale, front_b):
    x = Fb[i0]
    dominated = bool(((Fb <= x).all(1) & (Fb < x).any(1)).any().item())
    d = (((Fb[torch.as_tensor(front_b, device=Fb.device)] - x) / scale) ** 2).sum(1).sqrt().min().item()
    return (not dominated), (0.0 if not dominated else d)


def ari(la, lb, device):
    a = torch.as_tensor(la, device=device)
    b = torch.as_tensor(lb, device=device)
    n = a.numel()
    _, ia = torch.unique(a, return_inverse=True)
    _, ib = torch.unique(b, return_inverse=True)
    ka, kb = int(ia.max()) + 1, int(ib.max()) + 1
    cont = torch.bincount(ia * kb + ib, minlength=ka * kb).to(F64).view(ka, kb)
    c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    tot = n * (n - 1) / 2
    exp = sa * sb / tot
    mx = (sa + sb) / 2
    if float(mx - exp) == 0.0:
        return 1.0
    return float((sij - exp) / (mx - exp))
