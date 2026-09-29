"""Path pool (Algorithm 3.3 output), valid graphoidal-cover assembly, candidate generation.

Numpy-only (no torch) so the combinatorial part is testable anywhere.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .dijkstra import fuzzy_dijkstra, label_to_path
from .tfn import TFNOps, rank_arrays


class PathPool:
    def __init__(self, graph, paths):
        """paths: list of (verts tuple, edges tuple), canonical orientation, unique."""
        self.verts = [p[0] for p in paths]
        self.edges = [p[1] for p in paths]
        self.internal = [v[1:-1] for v in self.verts]
        self.n_edges = np.array([len(e) for e in self.edges], dtype=int)
        P = len(paths)
        self.maxlen = int(self.n_edges.max())
        self.edge_pad = -np.ones((P, self.maxlen), dtype=np.int64)
        self.vert_pad = -np.ones((P, self.maxlen + 1), dtype=np.int64)
        for i, (v, e) in enumerate(zip(self.verts, self.edges)):
            self.edge_pad[i, :len(e)] = e
            self.vert_pad[i, :len(v)] = v
        Lm = graph.L
        # fuzzy addition of edge lengths (raw TFN lengths, not strength-scaled)
        self.sumL = np.array([Lm[list(e), 0].sum() for e in self.edges])
        self.sumM = np.array([Lm[list(e), 1].sum() for e in self.edges])
        self.sumU = np.array([Lm[list(e), 2].sum() for e in self.edges])
        self.single_id = -np.ones(graph.m, dtype=np.int64)
        for i, e in enumerate(self.edges):
            if len(e) == 1:
                self.single_id[e[0]] = i
        assert (self.single_id >= 0).all()
        self.multi_ids = np.where(self.n_edges >= 2)[0]

    def __len__(self):
        return len(self.edges)


def pick_sources(graph, n_sources):
    deg = graph.degrees()
    A = coo_matrix((np.ones(graph.m), (graph.edges[:, 0], graph.edges[:, 1])), shape=(graph.n, graph.n))
    _, comp = connected_components(A, directed=False)
    order = np.argsort(-deg, kind="stable")
    src = list(order[:n_sources])
    seen = {comp[s] for s in src}
    for v in order:                 # ensure every component has >=1 source
        if comp[v] not in seen:
            seen.add(comp[v])
            src.append(v)
    return [int(s) for s in src]


def build_pool(graph, cfg):
    ops = TFNOps(cfg.rank_mode)
    adj = graph.adjacency(cfg.strength_mode)
    seen, paths = set(), []
    for s in pick_sources(graph, cfg.n_sources):
        for lab in fuzzy_dijkstra(adj, s, ops, cfg.max_labels):
            if lab.parent is None:
                continue
            v, e = label_to_path(lab)
            if v[0] > v[-1]:
                v, e = v[::-1], e[::-1]
            if e in seen:
                continue
            seen.add(e)
            paths.append((v, e))
    for k, (i, j) in enumerate(graph.edges):          # trivial one-edge paths (always feasible)
        e = (k,)
        if e not in seen:
            seen.add(e)
            paths.append(((int(i), int(j)), e))
    return PathPool(graph, paths)


def assemble(pool, order, n_edges_total):
    """Greedy Step-4 replacement: accept a path only if all its edges are still free and none of
    its internal vertices is already internal to an accepted path; leftovers -> one-edge paths.
    Result is always a valid graphoidal cover."""
    used = np.zeros(n_edges_total, dtype=bool)
    internal_used = set()
    chosen = []
    for pid in order:
        es = pool.edges[pid]
        ok = True
        for e in es:
            if used[e]:
                ok = False
                break
        if not ok:
            continue
        iv = pool.internal[pid]
        if any(v in internal_used for v in iv):
            continue
        for e in es:
            used[e] = True
        internal_used.update(iv)
        chosen.append(pid)
    for e in np.where(~used)[0]:
        chosen.append(int(pool.single_id[e]))
    return np.array(chosen, dtype=np.int64)


def verify_cover(pool, cover, n_edges_total):
    """Check graphoidal-cover axioms; raises AssertionError on violation."""
    cnt = np.zeros(n_edges_total, dtype=int)
    internal = {}
    for pid in cover:
        for e in pool.edges[pid]:
            cnt[e] += 1
        for v in pool.internal[pid]:
            internal[v] = internal.get(v, 0) + 1
    assert (cnt == 1).all(), "an edge is not covered exactly once"
    assert all(c == 1 for c in internal.values()), "a vertex is internal to >1 path"
    return True


def path_conf_np(pool, mu_e, floor=1e-6):
    """s_r = (prod_e mu_E(e))^(1/|P_r|)  (geometric mean), numpy."""
    logm = np.log(np.clip(mu_e, floor, 1.0))
    pad = np.maximum(pool.edge_pad, 0)
    vals = np.where(pool.edge_pad >= 0, logm[pad], 0.0)
    return np.exp(vals.sum(1) / pool.n_edges)


def generate_covers(pool, graph, mu_e, cfg, n_orders, seed, cost):
    """Randomised greedy assemblies -> DISTINCT valid covers (the shared candidate set C).
    Random objective weights + random caps on path length / cost / relative uncertainty give
    covers ranging from compact (few long paths) to cheap/certain (many short, controlled paths)."""
    rng = np.random.default_rng(seed)
    conf = path_conf_np(pool, mu_e)
    m = pool.n_edges.astype(float)
    Ur = (pool.sumU - pool.sumL) / (pool.sumM + cfg.eps)
    len_n = m / m.max()
    unc_n = Ur / (Ur.max() + cfg.eps)
    cost_n = cost / (cost.max() + cfg.eps)
    multi = pool.multi_ids
    if len(multi) == 0:          # tiny graph: no path with >= 2 edges exists -> the only cover is the trivial one
        return [assemble(pool, multi, graph.m)]
    corners = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (.25, .25, .25, .25)]
    seen, covers = set(), []
    for t in range(n_orders):
        w = np.array(corners[t], dtype=float) if t < len(corners) else rng.dirichlet(np.ones(4))
        ids = multi
        if not (t < len(corners) or rng.random() < 0.3):
            cap = int(rng.integers(2, max(3, pool.maxlen + 1)))
            ids = ids[pool.n_edges[ids] <= cap]
        if t >= len(corners) and rng.random() < 0.6:
            q = rng.uniform(0.15, 1.0)
            ids = ids[cost[ids] <= np.quantile(cost[multi], q)]
        if t >= len(corners) and rng.random() < 0.4:
            q = rng.uniform(0.15, 1.0)
            ids = ids[Ur[ids] <= np.quantile(Ur[multi], q)]
        if len(ids) == 0:
            ids = multi[:0]
        score = -w[0] * len_n[ids] + w[1] * (1 - conf[ids]) + w[2] * unc_n[ids] + w[3] * cost_n[ids] \
            + 0.05 * rng.random(len(ids))
        order = ids[np.argsort(score, kind="stable")]
        cov = assemble(pool, order, graph.m)
        sig = tuple(sorted(cov.tolist()))
        if sig in seen:
            continue
        seen.add(sig)
        covers.append(cov)
    return covers


def cover_signature(pool, cover):
    return frozenset(pool.edges[p] for p in cover)


def cover_edge_labels(pool, cover, n_edges_total):
    lab = np.empty(n_edges_total, dtype=np.int64)
    for i, pid in enumerate(cover):
        lab[list(pool.edges[pid])] = i
    return lab
