"""Offline smoke-test data with the same interface as data.load_brca (NOT biological data)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import BRCAData


def make_synthetic(G=900, n=300, module=18, seed=0):
    rng = np.random.default_rng(seed)
    n_mod = G // module
    latent = rng.normal(size=(n_mod, n))
    X = rng.normal(size=(G, n))
    mod = np.arange(G) // module
    strength = rng.uniform(0.3, 1.5, size=G)
    X = X + strength[:, None] * latent[np.minimum(mod, n_mod - 1)]
    X = np.exp(0.5 * X).astype(np.float32)
    genes = [f"G{i:04d}" for i in range(G)]
    rows = []
    for m in range(n_mod):
        idx = np.arange(m * module, (m + 1) * module)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if rng.random() < 0.35:
                    rows.append((genes[idx[a]], genes[idx[b]], rng.uniform(0.7, 0.999)))
    for _ in range(G // 2):
        a, b = rng.integers(0, G, 2)
        if a != b:
            lo, hi = sorted((genes[a], genes[b]))
            rows.append((lo, hi, rng.uniform(0.7, 0.95)))
    ed = pd.DataFrame(rows, columns=["a", "b", "score"]).groupby(["a", "b"], as_index=False).max()
    A = rng.beta(0.6, 4, size=G)
    A /= A.max()
    E = np.minimum(np.abs(rng.normal(0, 0.4, size=G)), 1.0)
    mu_v = 0.5 * A + 0.5 * E
    return BRCAData(genes, np.log2(X + 1), A, E, mu_v, ed, {"synthetic": True})
