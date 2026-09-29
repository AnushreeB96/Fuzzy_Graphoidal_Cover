"""Fuzzy graph container + GPU bootstrap construction of triangular fuzzy edge lengths (Sec. 7.3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FuzzyGraph:
    names: list            # vertex names (gene symbols)
    edges: np.ndarray      # (E,2) int, i<j
    mu_e: np.ndarray       # (E,) edge membership
    L: np.ndarray          # (E,3) triangular fuzzy lengths
    mu_v: np.ndarray       # (n,) vertex membership

    @property
    def n(self):
        return len(self.names)

    @property
    def m(self):
        return len(self.edges)

    def with_mu_e(self, mu_e):
        return FuzzyGraph(self.names, self.edges, np.asarray(mu_e), self.L, self.mu_v)

    def adjacency(self, strength_mode="none", floor=0.05):
        """adj[v] = [(w, tfn, edge_id)].  strength_mode='inverse' divides the fuzzy length by
        the edge membership (weak edges become 'longer') -> distance AND strength (assumption A2)."""
        adj = [[] for _ in range(self.n)]
        for e, (i, j) in enumerate(self.edges):
            l = self.L[e]
            if strength_mode == "inverse":
                s = 1.0 / max(float(self.mu_e[e]), floor)
                l = l * s
            t = (float(l[0]), float(l[1]), float(l[2]))
            adj[i].append((int(j), t, e))
            adj[j].append((int(i), t, e))
        return adj

    def degrees(self):
        d = np.zeros(self.n, dtype=int)
        np.add.at(d, self.edges[:, 0], 1)
        np.add.at(d, self.edges[:, 1], 1)
        return d


def _avg_rank(x):
    """Average (tie-aware) ranks along the last dim, fully on device."""
    import torch
    s, order = torch.sort(x, dim=-1)
    lo = torch.searchsorted(s, s, right=False)
    hi = torch.searchsorted(s, s, right=True)
    avg = (lo + hi - 1).to(x.dtype) / 2.0
    r = torch.empty_like(x)
    r.scatter_(-1, order, avg)
    return r


def bootstrap_edge_lengths(X, edges, B, device, seed=0, mem_budget=1.5e9):
    """X: (G,n) expression (genes x patients).  For each of B bootstrap resamples of patients,
    compute Spearman correlation of every gene pair (batched rank + bmm on GPU), then
    d = 1-|rho|.  Returns (E,3) array of the 10/50/90th percentiles = TFN (l,m,u)."""
    import torch
    G, n = X.shape
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    ei = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
    ej = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    chunk = max(1, int(mem_budget / (6 * 4 * G * n)))
    out = []
    for s in range(0, B, chunk):
        bc = min(chunk, B - s)
        idx = torch.randint(0, n, (bc, n), device=device, generator=gen)
        Xb = Xt[:, idx].permute(1, 0, 2).contiguous()          # (bc,G,n)
        R = _avg_rank(Xb)
        R = R - R.mean(-1, keepdim=True)
        R = R / (R.norm(dim=-1, keepdim=True) + 1e-12)
        C = torch.bmm(R, R.transpose(1, 2))                    # (bc,G,G) Spearman matrices
        out.append(1.0 - C[:, ei, ej].abs())                   # (bc,E)
        del Xb, R, C
    D = torch.cat(out, 0)                                      # (B,E)
    Ds, _ = torch.sort(D, dim=0)
    res = []
    for q in (0.10, 0.50, 0.90):
        pos = q * (B - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        fr = pos - lo
        res.append(Ds[lo] * (1 - fr) + Ds[hi] * fr)
    Q = torch.stack(res, 1).cpu().numpy().astype(np.float64)   # (E,3)
    Q = np.maximum(Q, 1e-6)
    Q.sort(axis=1)                                             # enforce l<=m<=u
    return Q


def build_fuzzy_graph(data, N, cfg, device):
    """Sec. 7.4: top-N genes by vertex membership -> STRING edges (>=threshold) -> fuzzy graph."""
    order = np.argsort(-data.mu_v, kind="stable")
    top = order[:N]
    gset = {data.genes[i]: k for k, i in enumerate(top)}
    ed = data.string_edges
    keep = ed["a"].isin(gset) & ed["b"].isin(gset)
    ed = ed[keep]
    ia = ed["a"].map(gset).to_numpy()
    ib = ed["b"].map(gset).to_numpy()
    sc = ed["score"].to_numpy(dtype=float)
    # drop isolated vertices
    used = np.unique(np.concatenate([ia, ib]))
    remap = -np.ones(len(top), dtype=int)
    remap[used] = np.arange(len(used))
    sel_genes = top[used]
    names = [data.genes[i] for i in sel_genes]
    edges = np.stack([remap[ia], remap[ib]], 1)
    edges.sort(axis=1)
    mu_v = data.mu_v[sel_genes]
    mu_e = sc * np.sqrt(mu_v[edges[:, 0]] * mu_v[edges[:, 1]])
    if cfg.clip_edge_membership:
        mu_e = np.minimum(mu_e, np.minimum(mu_v[edges[:, 0]], mu_v[edges[:, 1]]))
    mu_e = np.clip(mu_e, 0.0, 1.0)
    L = bootstrap_edge_lengths(data.expr[sel_genes], edges, cfg.n_boot, device, seed=cfg.seed)
    return FuzzyGraph(names, edges, mu_e, L, mu_v)
