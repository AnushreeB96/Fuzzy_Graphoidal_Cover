"""Paired statistics across seeds: Wilcoxon signed-rank, rank-biserial r, Cohen's dz, Holm correction."""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, rankdata


def paired_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(n=len(d), mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean())
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        return {**out, "W": 0.0, "p": 1.0, "rank_biserial": 0.0, "cohen_dz": 0.0}
    ranks = rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    try:
        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {**out, "W": float(min(wp, wm)), "p": p, "rank_biserial": float((wp - wm) / (wp + wm)),
            "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan")}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
