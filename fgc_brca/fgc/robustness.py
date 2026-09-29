"""Sec. 6 + review: Monte-Carlo perturbation of memberships AND fuzzy lengths with FOUR complementary stability measures.

All methods re-select from the SAME fixed candidate set under each perturbed membership vector:
  * structural stability  : exact-match rate and adjusted Rand index (ARI) of edge partitions
  * path-count stability  : |k_b - k_0|
  * objective regret      : (f_j(Psi_b) - f_j(Psi_0)) / (|f_j(Psi_0)| + eps),  j = 1..4
  * Pareto-front stability: is the nominal choice Psi_0 still non-dominated after perturbation, and how far
                            (normalised Euclidean) is it from the perturbed front
"""
from __future__ import annotations
import numpy as np
import torch

from .objectives import ari, nondominated_and_distance
from .pipeline import select_methods, perturb_data, edge_labels, METHODS


def robustness_experiment(graph, cands, cfg, device, log=print):
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 1)
    E = graph.m
    base = select_methods(cands, graph.mu_e, graph.L, cfg, device)
    F0 = base["F"]
    scale = (F0.max(0).values - F0.min(0).values) + cfg.eps
    F0n = F0.cpu().numpy()
    rows, per_real, fronts = [], [], {}
    TIE_CAP = 50
    for delta in cfg.deltas:
        B = cfg.mc_B if delta > 0 else 1
        rec = {m: {k: [] for k in ["exact", "ari", "ari_tie", "nties", "dk", "dcost", "r1", "r2", "r3", "r4", "nondom", "pdist"]} for m in METHODS}
        fsizes = []
        for b in range(B):
            mu_b, L_b = perturb_data(graph.mu_e, graph.L, delta, 1, gen, device)   # memberships AND lengths
            res = select_methods(cands, mu_b[0], L_b[0], cfg, device)
            Fb, Fbn = res["F"], res["F"].cpu().numpy()
            fsizes.append(len(res["front"]))
            if b == 0:
                fronts[delta] = Fbn[res["front"]]
            for m in METHODS:
                i0, ib = base["idx"][m], res["idx"][m]
                r = rec[m]
                r["exact"].append(float(i0 == ib))
                a_main = ari(edge_labels(cands, ib, E), edge_labels(cands, i0, E), device)
                r["ari"].append(a_main)
                # tie-aware ARI: best agreement with ANY of the equally-optimal nominal covers
                nom_ties = [t for t in base["ties"][m][:TIE_CAP]]
                a_tie = a_main if len(nom_ties) <= 1 else max(
                    a_main, max(ari(edge_labels(cands, ib, E), edge_labels(cands, int(t), E), device) for t in nom_ties))
                r["ari_tie"].append(a_tie)
                r["nties"].append(len(res["ties"][m]))
                per_real.append(dict(delta=delta, b=b, method=m, selected_id=ib, nominal_id=i0,
                                     n_equally_optimal=len(res["ties"][m]), ari=a_main, ari_tie_aware=a_tie))
                r["dk"].append(abs(cands.k[ib].item() - cands.k[i0].item()))
                r["dcost"].append(abs(Fbn[ib, 1] - F0n[i0, 1]) / (abs(F0n[i0, 1]) + cfg.eps))
                for j in range(4):
                    r[f"r{j + 1}"].append((Fbn[ib, j] - F0n[i0, j]) / (abs(F0n[i0, j]) + cfg.eps))
                nd, dist = nondominated_and_distance(Fb, i0, scale, res["front"])
                r["nondom"].append(float(nd)); r["pdist"].append(dist)
            if (b + 1) % 25 == 0:
                log(f"   delta={delta:.2f}: {b + 1}/{B}")
        for m in METHODS:
            row = dict(delta=delta, method=m, mean_front_size=float(np.mean(fsizes)))
            for k, v in rec[m].items():
                row[f"{k}_mean"], row[f"{k}_sd"] = float(np.mean(v)), float(np.std(v))
            rows.append(row)
    return rows, fronts, base, per_real
