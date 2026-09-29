"""Candidate generation (once) + method selection (repeatable under perturbed data)."""
from __future__ import annotations
import time
import numpy as np
import torch

from .cover import build_pool, generate_covers, verify_cover, cover_edge_labels
from .objectives import Candidates, path_stats, pareto_front, compromise_select
from .tfn import rank_arrays

METHODS = ("min_card", "min_wc", "proposed", "robust")
METHOD_NAMES = {"min_card": "Minimum-cardinality", "min_wc": "Minimum worst-cost",
                "proposed": "Proposed Pareto", "robust": "Robustness-aware Pareto"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def n_candidates_for(cfg, n_requested):
    return cfg.n_candidates if cfg.n_candidates else cfg.cand_by_size.get(n_requested, 1000)


def build_candidates(graph, cfg, device, n_orders):
    """Shared candidate set C: generated ONCE on the nominal graph and used by every method."""
    t0 = time.perf_counter()
    pool = build_pool(graph, cfg)
    cost = rank_arrays(pool.sumL, pool.sumM, pool.sumU, cfg.rank_mode)
    covers = generate_covers(pool, graph, graph.mu_e, cfg, n_orders, cfg.seed, cost)
    cands = Candidates(pool, covers, cfg, device)
    sync(device)
    return cands, time.perf_counter() - t0


def perturb_data(mu_e, L, delta, B, gen, device):
    """Perturb memberships AND fuzzy lengths, B independent realisations.
      mu' = clip(mu + U(-delta, delta), 0, 1)
      L'  = L * (1 + U(-delta, delta))  independently for l, m, u; then sorted so l' <= m' <= u'.
    Returns mu (B,E), L (B,E,3) float64 tensors."""
    mu = torch.as_tensor(mu_e, dtype=torch.float64, device=device)[None].expand(B, -1)
    Lt = torch.as_tensor(L, dtype=torch.float64, device=device)[None].expand(B, -1, -1)
    if delta == 0:
        return mu.clone(), Lt.clone()
    r = lambda shape: (torch.rand(shape, generator=gen, device=device, dtype=torch.float64) * 2 - 1) * delta
    mu_p = torch.clamp(mu + r(mu.shape), 0.0, 1.0)
    L_p = (Lt * (1 + r(Lt.shape))).clamp(min=1e-6)
    L_p, _ = torch.sort(L_p, dim=-1)
    return mu_p, L_p


def select_methods(cands, mu_e, L, cfg, device):
    """Evaluate all candidates on the given memberships/lengths and choose one cover per method."""
    pt, T = cands.pt, {}
    t0 = time.perf_counter()
    mu = torch.as_tensor(mu_e, dtype=torch.float64, device=device)[None]
    Lb = torch.as_tensor(L, dtype=torch.float64, device=device)[None]
    F, mU = cands.objectives(*path_stats(pt, mu, Lb))
    F, mU = F[0], mU[0]
    Fn = F.cpu().numpy()
    # Deterministic tie-breaking (documented): min_card = lexicographic (f1,f2,f3,f4);
    # min_wc = lexicographic (f2,f1,f3,f4).  `ties[m]` = all candidates tied at the primary criterion.
    o_card = np.lexsort((Fn[:, 3], Fn[:, 2], Fn[:, 1], Fn[:, 0]))
    o_wc = np.lexsort((Fn[:, 3], Fn[:, 2], Fn[:, 0], Fn[:, 1]))
    idx = {"min_card": int(o_card[0]), "min_wc": int(o_wc[0])}
    ties = {}
    for m, o, col in (("min_card", o_card, 0), ("min_wc", o_wc, 1)):
        thr = Fn[o[0], col] + 1e-9
        k_ = 0
        while k_ < len(o) and Fn[o[k_], col] <= thr:
            k_ += 1
        ties[m] = o[:k_]
    sync(device); T["eval"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    front = pareto_front(F)
    li, lt = compromise_select(F[torch.as_tensor(front, device=device)], cfg.lam, cfg.eps)
    idx["proposed"], ties["proposed"] = int(front[li]), front[lt]
    sync(device); T["proposed"] = time.perf_counter() - t0

    # robust: ensemble of robust_B runs at EACH level in robust_deltas (pooled); f^R = mean + kappa*SD
    t0 = time.perf_counter()
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 7)                       # common random numbers -> deterministic
    mus, Ls = [], []
    for d in cfg.robust_deltas:
        m_, l_ = perturb_data(mu_e, L, d, cfg.robust_B, gen, device)
        mus.append(m_); Ls.append(l_)
    Fin, _ = cands.objectives(*path_stats(pt, torch.cat(mus), torch.cat(Ls)))   # (Btot,C,4)
    FR = Fin.mean(0) + cfg.robust_kappa * Fin.std(0)
    frontR = pareto_front(FR)
    li, lt = compromise_select(FR[torch.as_tensor(frontR, device=device)], cfg.lam, cfg.eps)
    idx["robust"], ties["robust"] = int(frontR[li]), frontR[lt]
    del Fin
    sync(device); T["robust"] = time.perf_counter() - t0
    for m in METHODS:
        verify_cover(cands.pool, cands.covers[idx[m]], cands.pool.single_id.shape[0])
    return dict(F=F, meanU=mU, FR=FR, front=front, frontR=frontR, idx=idx, ties=ties, time=T)


def edge_labels(cands, i, n_edges):
    if i not in cands.labels_cache:
        cands.labels_cache[i] = cover_edge_labels(cands.pool, cands.covers[i], n_edges)
    return cands.labels_cache[i]
