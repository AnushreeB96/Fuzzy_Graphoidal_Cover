"""CPU/numpy-only tests of the combinatorial core (no torch needed):  python tests_numpy.py"""
import numpy as np
from types import SimpleNamespace
from fgc.tfn import TFNOps, rank_arrays
from fgc.graph import FuzzyGraph
from fgc.cover import build_pool, generate_covers, verify_cover

ops = TFNOps("yager")
assert ops.totally_governs((1, 2, 3), (1, 3, 4)) and not ops.totally_governs((1, 4, 5), (1, 3, 4))
assert not ops.governs((5, 5, 5), (0, 0, 100)) and not ops.governs((0, 0, 100), (5, 5, 5))
assert ops.least((1, 2, 3), (2, 3, 4)) == (1, 2, 3)

rng = np.random.default_rng(0)
n = 80
E = set()
while len(E) < 220:
    a, b = sorted(rng.integers(0, n, 2))
    if a != b:
        E.add((int(a), int(b)))
E = np.array(sorted(E))
m = rng.uniform(0.1, 0.9, len(E)); w = rng.uniform(0.05, 0.3, len(E))
L = np.stack([m - w * .5, m, m + w], 1)
g = FuzzyGraph([str(i) for i in range(n)], E, rng.uniform(0.2, 1, len(E)), L, rng.uniform(0.2, 1, n))
cfg = SimpleNamespace(rank_mode="yager", strength_mode="none", n_sources=10, max_labels=4, eps=1e-9)
pool = build_pool(g, cfg)
cost = rank_arrays(pool.sumL, pool.sumM, pool.sumU, "yager")
covs = generate_covers(pool, g, g.mu_e, cfg, 150, 1, cost)
for c in covs:
    verify_cover(pool, c, g.m)
k = np.array([len(c) for c in covs])
wc = np.array([cost[c].max() for c in covs])
Ur = (pool.sumU - pool.sumL) / (pool.sumM + 1e-9)
wu = np.array([Ur[c].max() for c in covs])
floor = cost[pool.single_id].max()
print(f"{len(covs)} distinct valid covers | k in [{k.min()},{k.max()}] | worst cost in [{wc.min():.3f},{wc.max():.3f}] "
      f"(floor = max edge cost {floor:.3f}) | worst unc in [{wu.min():.3f},{wu.max():.3f}]")
f3 = np.array([0.7 * Ur[c].mean() + 0.3 * np.quantile(Ur[c], 0.90) for c in covs])
print(f"f3 (0.7*mean u + 0.3*Q90 u) in [{f3.min():.4f},{f3.max():.4f}], distinct values: {len(set(np.round(f3, 9)))}")
assert len(set(np.round(f3, 9))) > 3, "f3 must discriminate between covers"
assert wc.min() >= floor - 1e-9
assert len(set(np.round(wc, 9))) > 3 and len(set(k)) > 3, "objectives must discriminate between covers"
print("ALL OK")
