"""Figures: Fig.1 Pareto front, Fig.2 biological analysis."""
from __future__ import annotations
import numpy as np


def fig_pareto(F, front, marks, path_png, n_total_label=""):
    """F: (C,4) nominal objectives; front: non-dominated indices; marks: {label: candidate index}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, col, lab, cmap in ((ax[0], -F[:, 3], "mean confidence", "viridis"),
                              (ax[1], F[:, 2], "uncertainty f3 (0.7·mean + 0.3·Q90)", "magma_r")):
        a.scatter(F[:, 0], F[:, 1], s=8, c="lightgray", label="all candidates", zorder=1)
        sc = a.scatter(F[front, 0], F[front, 1], c=col[front], cmap=cmap, s=40, edgecolor="k", linewidth=.3,
                       label="non-dominated", zorder=2)
        plt.colorbar(sc, ax=a, label=lab)
        for (name, i), mk in zip(marks.items(), ["*", "P", "X", "D", "s", "^"]):
            a.scatter(F[i, 0], F[i, 1], marker=mk, s=170, c="red", edgecolor="k", zorder=5, label=name)
        a.set_xlabel("number of paths (f1)"); a.set_ylabel("worst-path fuzzy cost (f2)")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title(f"Pareto front - confidence {n_total_label}")
    ax[1].set_title("Pareto front - uncertainty")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.savefig(path_png.replace(".png", ".pdf"))
    plt.close(fig)


def fig_biology(g, cands, cov, s_path, cover_hits, path_png, top=10):
    """Left: network of genes in multi-gene paths coloured by path; right: top Reactome pathways (cover level)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    pool = cands.pool
    multi = [p for p in cov if pool.n_edges[p] >= 2]
    try:
        import networkx as nx
        G = nx.Graph()
        cmap = plt.get_cmap("tab20")
        for ci, p in enumerate(multi):
            vs = pool.verts[p]
            for a, b in zip(vs[:-1], vs[1:]):
                G.add_edge(g.names[a], g.names[b], color=cmap(ci % 20))
        pos = nx.spring_layout(G, seed=1, k=0.6)
        nx.draw_networkx_edges(G, pos, ax=ax[0], edge_color=[G[u][v]["color"] for u, v in G.edges], width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax[0], node_size=60, node_color="white", edgecolors="k")
        deg = dict(G.degree())
        lab = {n: n for n in sorted(deg, key=deg.get, reverse=True)[:25]}
        nx.draw_networkx_labels(G, pos, labels=lab, ax=ax[0], font_size=6)
        ax[0].set_title(f"Selected balanced cover: {len(multi)} multi-gene paths (colour = path)")
    except Exception as ex:
        ax[0].text(0.1, 0.5, f"network drawing unavailable: {ex}")
    ax[0].axis("off")
    if cover_hits:
        hits = [h for h in cover_hits if h.get("fdr") is not None][:top]
        names = [h["name"][:55] for h in hits][::-1]
        vals = [-np.log10(max(h["fdr"], 1e-300)) for h in hits][::-1]
        ax[1].barh(names, vals, color="C0")
        ax[1].axvline(-np.log10(0.05), c="r", ls="--", lw=1, label="FDR = 0.05")
        ax[1].set_xlabel("-log10(FDR)"); ax[1].set_title("Reactome over-representation (genes of the cover)")
        ax[1].legend(fontsize=7); ax[1].tick_params(axis="y", labelsize=7)
    else:
        ax[1].text(0.1, 0.5, "Reactome results unavailable (skipped or offline)"); ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.savefig(path_png.replace(".png", ".pdf"))
    plt.close(fig)
