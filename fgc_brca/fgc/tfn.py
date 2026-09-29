"""Triangular fuzzy numbers (TFNs): arithmetic, ranking, governing relations, LEAST.

A TFN is a tuple (l, m, u) with l <= m <= u.

The manuscript's Sec. 2.3 (ranking function, 'totally governing', 'fractionally
governing') was not supplied, so the definitions below are explicit, documented
assumptions (README, A1).  Swap them here if your Sec. 2.3 differs.
"""
from __future__ import annotations
import numpy as np

TOL = 1e-12


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def rank_scalar(a, mode="yager"):
    l, m, u = a
    if mode == "yager":        # (l+2m+u)/4  (linear)
        return (l + 2 * m + u) / 4.0
    if mode == "centroid":     # (l+m+u)/3   (linear)
        return (l + m + u) / 3.0
    if mode == "risk":         # yager + 0.25*(u-l): penalises uncertain numbers (linear)
        return (l + 2 * m + u) / 4.0 + 0.25 * (u - l)
    raise ValueError(f"unknown rank mode {mode}")


def rank_arrays(L, M, U, mode="yager"):
    L, M, U = map(np.asarray, (L, M, U))
    if mode == "yager":
        return (L + 2 * M + U) / 4.0
    if mode == "centroid":
        return (L + M + U) / 3.0
    if mode == "risk":
        return (L + 2 * M + U) / 4.0 + 0.25 * (U - L)
    raise ValueError(mode)


class TFNOps:
    """Bundle of TFN relations for a chosen ranking mode."""

    def __init__(self, rank_mode="yager"):
        self.rank_mode = rank_mode

    def rank(self, a):
        return rank_scalar(a, self.rank_mode)

    def totally_governs(self, a, b):
        """a is totally governing b  (a <=_t b):  every defining point of a <= that of b."""
        return a[0] <= b[0] + TOL and a[1] <= b[1] + TOL and a[2] <= b[2] + TOL

    def fractionally_governs(self, a, b):
        """a is fractionally governing b (a <_f b): not totally governing, but at least two of
        the three defining points are <= and the ranking value of a does not exceed that of b."""
        if self.totally_governs(a, b):
            return False
        cnt = (a[0] <= b[0] + TOL) + (a[1] <= b[1] + TOL) + (a[2] <= b[2] + TOL)
        return cnt >= 2 and self.rank(a) <= self.rank(b) + TOL

    def governs(self, a, b):
        return self.totally_governs(a, b) or self.fractionally_governs(a, b)

    def least(self, q, r):
        """FUNCTION LEAST(Q, R) of the manuscript."""
        if self.totally_governs(q, r) or self.fractionally_governs(q, r):
            return q
        return r


def rank_tensors(S, mode="yager"):
    """Torch version of the ranking function; S[..., 3] = (l, m, u)."""
    l, m, u = S[..., 0], S[..., 1], S[..., 2]
    if mode == "yager":
        return (l + 2 * m + u) / 4.0
    if mode == "centroid":
        return (l + m + u) / 3.0
    if mode == "risk":
        return (l + 2 * m + u) / 4.0 + 0.25 * (u - l)
    raise ValueError(mode)
