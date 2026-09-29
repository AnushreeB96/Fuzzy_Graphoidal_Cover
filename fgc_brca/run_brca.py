#!/usr/bin/env python
"""Multi-objective fuzzy graphoidal cover on TCGA-BRCA (GPU): reproducibility rerun with multiple seeds.

  python run_brca.py                       # full experiment (10 seeds; needs internet on first run)
  python run_brca.py --synthetic --quick   # offline smoke test

Frozen formulation:  f1=k, f2=max_r R(L(P_r)), f3=0.7*mean u + 0.3*Q90(u), f4=-mean confidence.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch

from fgc.config import Config, PREFERENCES
from fgc.graph import build_fuzzy_graph
from fgc.pipeline import build_candidates, select_methods, n_candidates_for, METHODS, METHOD_NAMES
from fgc.objectives import compromise_select
from fgc.robustness import robustness_experiment
from fgc.cover import path_conf_np
from fgc.stats import paired_test, holm
from fgc import report


def parse():
    c = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=c.outdir)
    p.add_argument("--data_dir", default=c.data_dir)
    p.add_argument("--device", default=c.device, choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=c.seed, help="first seed; seeds = seed, seed+1, ...")
    p.add_argument("--n_seeds", type=int, default=c.n_seeds)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no_resume", action="store_true", help="ignore existing per-seed checkpoints")
    p.add_argument("--skip_reactome", action="store_true")
    p.add_argument("--reactome_max_paths", type=int, default=c.reactome_max_paths)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--alpha", type=float, default=c.alpha)
    p.add_argument("--string_threshold", type=float, default=c.string_threshold)
    p.add_argument("--n_boot", type=int, default=c.n_boot)
    p.add_argument("--clip_edge_membership", action="store_true")
    p.add_argument("--main_size", type=int, default=c.main_size)
    p.add_argument("--sizes", type=int, nargs="+", default=list(c.sizes))
    p.add_argument("--n_sources", type=int, default=c.n_sources)
    p.add_argument("--max_labels", type=int, default=c.max_labels)
    p.add_argument("--strength_mode", default=c.strength_mode, choices=["none", "inverse"])
    p.add_argument("--rank_mode", default=c.rank_mode, choices=["yager", "centroid", "risk"])
    p.add_argument("--n_candidates", type=int, default=None)
    p.add_argument("--lam", type=float, nargs=4, default=list(c.lam))
    p.add_argument("--robust_deltas", type=float, nargs="+", default=list(c.robust_deltas))
    p.add_argument("--robust_B", type=int, default=c.robust_B)
    p.add_argument("--robust_kappa", type=float, default=c.robust_kappa)
    p.add_argument("--mc_B", type=int, default=c.mc_B)
    p.add_argument("--deltas", type=float, nargs="+", default=list(c.deltas))
    p.add_argument("--stability_delta", type=float, default=c.stability_delta)
    a = p.parse_args()
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in vars(a).items() if k not in ("quick", "no_resume")})
    cfg.resume = not a.no_resume
    if a.quick:
        cfg.n_boot, cfg.mc_B, cfg.robust_B, cfg.n_seeds = 50, 10, 20, 2
        cfg.cand_by_size = {50: 60, 100: 100, 200: 100, 500: 200}
        cfg.sizes, cfg.main_size = (50, 100, 200), min(cfg.main_size, 100)
    return cfg


def mem_report(device):
    import psutil
    cpu = psutil.Process().memory_info().rss / 2**20
    gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return cpu, gpu


def pm(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {(x.std(ddof=1) if len(x) > 1 else 0.0):.{nd}f}"


def md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".4g")
    except Exception:
        return df.to_string(index=False)


# --------------------------------------------------------------------------------------------
def run_seed(data, cfg, N, seed, device, log, ckpt=None):
    c = dataclasses.replace(cfg, seed=seed)
    t0 = time.perf_counter()
    g = build_fuzzy_graph(data, N, c, device)
    t_build = time.perf_counter() - t0
    cands, t_gen = build_candidates(g, c, device, n_candidates_for(c, N))
    sel = select_methods(cands, g.mu_e, g.L, c, device)
    if ckpt is not None:            # first seed resumed: rebuild objects for figures, keep saved statistics
        rob, per, fronts, base = ckpt["rob"], ckpt["per"], ckpt["fronts"], None
    else:
        rows, fronts, base, per = robustness_experiment(g, cands, c, device, log=log)
        rob = pd.DataFrame(rows)
    F = sel["F"].cpu().numpy()
    mU = sel["meanU"].cpu().numpy()
    T = sel["time"]
    extra = {"min_card": 0.0, "min_wc": 0.0, "proposed": T["proposed"], "robust": T["robust"]}
    recs = []
    for m in METHODS:
        i = sel["idx"][m]
        r = dict(seed=seed, method=m, paths=int(F[i, 0]), worst_cost=F[i, 1], confidence=-F[i, 3],
                 uncertainty=float(mU[i]), f3=F[i, 2], runtime=t_gen + T["eval"] + extra[m],
                 n_equally_optimal_nominal=len(sel["ties"][m]), selected_id=i)
        for d in cfg.deltas:
            if d == 0:
                continue
            x = rob[(rob.method == m) & np.isclose(rob.delta, d)].iloc[0]
            k = int(round(d * 100))
            r[f"ari{k}"], r[f"ari_tie{k}"], r[f"dk{k}"], r[f"dcost{k}"] = x.ari_mean, x.ari_tie_mean, x.dk_mean, x.dcost_mean
            r[f"nondom{k}"], r[f"pdist{k}"] = x.nondom_mean, x.pdist_mean
        recs.append(r)
    # preference scenarios on the nominal front
    front = sel["front"]
    Ft = sel["F"][torch.as_tensor(front, device=device)]
    prefs = []
    for name, lam in PREFERENCES.items():
        i = int(front[compromise_select(Ft, lam, cfg.eps)[0]])
        prefs.append(dict(seed=seed, preference=name, paths=int(F[i, 0]), worst_cost=F[i, 1],
                          uncertainty_f3=F[i, 2], mean_uncertainty=float(mU[i]), confidence=-F[i, 3], cand=i))
    if ckpt is not None:
        recs, prefs = ckpt["recs"], ckpt["prefs"]
    return dict(g=g, cands=cands, sel=sel, F=F, mU=mU, rob=rob, per=per, fronts=fronts, base=base,
                recs=recs, prefs=prefs, t_gen=t_gen, t_build=t_build)


def fingerprint(cfg):
    keys = ["main_size", "mc_B", "robust_B", "robust_kappa", "robust_deltas", "deltas", "lam", "n_boot", "n_candidates",
            "cand_by_size", "synthetic", "alpha", "string_threshold", "rank_mode", "strength_mode", "n_sources",
            "max_labels", "clip_edge_membership"]
    d = dataclasses.asdict(cfg)
    return json.dumps({k: d[k] for k in keys}, sort_keys=True, default=str)


def bio_analysis(g, cands, idx, name, cfg, outdir):
    """Reactome on the multi-gene paths of ONE fixed cover (run after optimisation)."""
    from fgc.reactome import enrich
    pool, pt = cands.pool, cands.pt
    cov = cands.covers[idx]
    s_path = path_conf_np(pool, g.mu_e)
    multi = sorted([p for p in cov if pool.n_edges[p] >= 2], key=lambda p: (-pool.n_edges[p], -s_path[p]))
    rows = []
    for r, p in enumerate(multi[:cfg.reactome_max_paths], 1):
        genes = [g.names[v] for v in pool.verts[p]]
        hits = enrich(genes)
        top = hits[0] if hits else {}
        rows.append({"Path": f"P{r}", "Genes": "-".join(genes), "Number of genes": len(genes),
                     "Confidence": float(s_path[p]),
                     "Fuzzy length": f"({pool.sumL[p]:.3f}, {pool.sumM[p]:.3f}, {pool.sumU[p]:.3f})",
                     "Fuzzy cost": float(pt.cost_np[p]), "Uncertainty": float(pt.Ur_np[p]),
                     "Reactome pathway": top.get("name", "n/a"), "Adjusted p-value (FDR)": top.get("fdr", np.nan)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, f"table_reactome_paths_{name}.csv"), index=False)
    allg = sorted({g.names[v] for p in cov if pool.n_edges[p] >= 2 for v in pool.verts[p]})
    hits = enrich(allg)
    pd.DataFrame(hits).to_csv(os.path.join(outdir, f"reactome_cover_{name}.csv"), index=False)
    fd = tab["Adjusted p-value (FDR)"].astype(float)
    summ = dict(method=METHOD_NAMES[name], multi_gene_paths=len(multi), paths_enriched=len(tab),
                paths_with_FDR_lt_0_05=int((fd < 0.05).sum()),
                fraction_significant=float((fd < 0.05).mean()) if len(fd) else np.nan,
                median_best_FDR=float(np.nanmedian(fd)) if len(fd) else np.nan,
                genes_in_multigene_paths=len(allg),
                cover_level_significant_pathways=int(sum(1 for h in hits if h.get("fdr") is not None and h["fdr"] < 0.05)))
    return summ, hits


# --------------------------------------------------------------------------------------------
def main():
    cfg = parse()
    device = torch.device("cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  (no GPU found - CPU)"))
    os.makedirs(cfg.outdir, exist_ok=True)
    out = lambda f: os.path.join(cfg.outdir, f)
    json.dump(dataclasses.asdict(cfg), open(out("config.json"), "w"), indent=2, default=str)

    if cfg.synthetic:
        from fgc.synthetic import make_synthetic
        data = make_synthetic(seed=cfg.seed)
    else:
        from fgc.data import load_brca
        data = load_brca(cfg)
    json.dump(data.info, open(out("data_info.json"), "w"), indent=2, default=str)

    # =============================================== Table 4: scalability (target vs actual genes)
    scal = []
    for N in sorted(set(cfg.sizes)):
        print(f"\n=== scalability: top-{N} selection ===")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        c = dataclasses.replace(cfg, seed=cfg.seed)
        t0 = time.perf_counter()
        g = build_fuzzy_graph(data, N, c, device)
        cands, t_gen = build_candidates(g, c, device, n_candidates_for(c, N))
        sel = select_methods(cands, g.mu_e, g.L, c, device)
        rt = time.perf_counter() - t0
        cpu, gpu = mem_report(device)
        budget = n_candidates_for(c, N)
        scal.append({"Target selection": f"Top-{N}", "Actual genes": g.n, "Edges": g.m,
                     "Candidate budget": budget, "Candidates (distinct)": cands.C,
                     "Budget cap reached": bool(cands.C >= budget), "Pareto solutions": len(sel["front"]),
                     "Runtime (s)": rt, "CPU mem (MB)": cpu, "GPU mem (MB)": gpu,
                     "Proposed cover size": int(cands.k[sel["idx"]["proposed"]].item()),
                     "Min-cardinality cover size": int(cands.k[sel["idx"]["min_card"]].item())})
        print("  ", scal[-1])
        del cands, sel
    scal = pd.DataFrame(scal)
    scal.to_csv(out("table4_scalability.csv"), index=False)

    # =============================================== multi-seed main experiment
    seeds = [cfg.seed + i for i in range(cfg.n_seeds)]
    results = {}
    ck_dir = out("checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    fp = fingerprint(cfg)
    for i, s in enumerate(seeds):
        path = os.path.join(ck_dir, f"seed_{s}.pkl")
        ck = None
        if cfg.resume and os.path.exists(path):
            try:
                ck = pickle.load(open(path, "rb"))
                if ck.get("fp") != fp:
                    print(f"[resume] checkpoint for seed {s} was made with different settings -> recomputing")
                    ck = None
            except Exception:
                ck = None
        print(f"\n=== main network top-{cfg.main_size}, seed {s} ({i + 1}/{len(seeds)}), B={cfg.mc_B} ===")
        if ck is not None and i > 0:
            print("[resume] seed already finished - loaded from checkpoint")
            results[s] = ck
            continue
        if ck is not None:
            print("[resume] seed finished earlier - rebuilding objects for figures only")
        results[s] = run_seed(data, cfg, cfg.main_size, s, device, log=lambda m: print(m), ckpt=ck)
        if ck is None:
            keep = {k: results[s][k] for k in ("recs", "prefs", "per", "rob", "fronts")}
            pickle.dump(dict(fp=fp, **keep), open(path, "wb"))
    seed_df = pd.DataFrame([r for s in seeds for r in results[s]["recs"]])
    seed_df.to_csv(out("seed_results.csv"), index=False)
    pref_df = pd.DataFrame([r for s in seeds for r in results[s]["prefs"]])
    pref_df.to_csv(out("seed_preferences.csv"), index=False)
    pd.DataFrame([dict(seed=s, **r) for s in seeds for r in results[s]["per"]]).to_csv(out("robustness_realizations.csv"), index=False)
    pd.concat([results[s]["rob"].assign(seed=s) for s in seeds]).to_csv(out("robustness_full_by_seed.csv"), index=False)

    ks = [int(round(d * 100)) for d in cfg.deltas if d > 0]
    # ------------------------------------------------ Table 1: main comparison (mean ± SD over seeds)
    t1 = []
    for m in METHODS:
        x = seed_df[seed_df.method == m]
        row = {"Method": METHOD_NAMES[m], "Paths ↓": pm(x.paths, 1), "Worst cost ↓": pm(x.worst_cost),
               "Confidence ↑": pm(x.confidence, 4), "Uncertainty ↓": pm(x.uncertainty, 4), "Runtime (s) ↓": pm(x.runtime, 2)}
        for k in ks:
            row[f"ARI ±{k}% ↑"] = pm(x[f"ari{k}"])
        t1.append(row)
    t1 = pd.DataFrame(t1)
    t1.to_csv(out("table1_main_comparison.csv"), index=False)

    # ------------------------------------------------ Table 2: preference sensitivity
    t2 = []
    for name in PREFERENCES:
        x = pref_df[pref_df.preference == name]
        t2.append({"Preference": name, "lambda (paths,cost,unc,-conf)": str(PREFERENCES[name]),
                   "Paths": pm(x.paths, 1), "Worst cost": pm(x.worst_cost), "Uncertainty": pm(x.mean_uncertainty, 4),
                   "Confidence": pm(x.confidence, 4)})
    t2 = pd.DataFrame(t2)
    t2.to_csv(out("table2_preference_sensitivity.csv"), index=False)
    # single-seed (main seed) detail
    pref_df[pref_df.seed == seeds[0]].to_csv(out("table2_preference_main_seed.csv"), index=False)

    # ------------------------------------------------ Table 3: robustness (secondary analysis)
    t3 = []
    ks_c = cfg.stability_delta and int(round(cfg.stability_delta * 100))
    for m in METHODS:
        x = seed_df[seed_df.method == m]
        row = {"Method": METHOD_NAMES[m]}
        for k in ks:
            row[f"ARI ±{k}%"] = pm(x[f"ari{k}"])
        for k in ks:
            row[f"Tie-aware ARI ±{k}%"] = pm(x[f"ari_tie{k}"])
        row[f"Δk ±{ks_c}%"] = pm(x[f"dk{ks_c}"], 2)
        row[f"Cost change ±{ks_c}%"] = pm(x[f"dcost{ks_c}"])
        row["Equally optimal covers (nominal)"] = pm(x.n_equally_optimal_nominal, 1)
        row[f"Non-dominated frac ±{ks_c}%"] = pm(x[f"nondom{ks_c}"], 2)
        t3.append(row)
    t3 = pd.DataFrame(t3)
    t3.to_csv(out("table3_robustness.csv"), index=False)

    # ------------------------------------------------ paired statistics (Proposed vs each other method)
    tests = []
    metrics = [("paths", "Paths"), ("worst_cost", "Worst cost"), ("confidence", "Confidence"),
               ("uncertainty", "Uncertainty"), ("runtime", "Runtime")] + [(f"ari{ks_c}", f"ARI ±{ks_c}%")]
    A = seed_df[seed_df.method == "proposed"].sort_values("seed")
    for other in ("min_card", "min_wc", "robust"):
        Bm = seed_df[seed_df.method == other].sort_values("seed")
        for col, lab in metrics:
            if len(A) < 2:
                continue
            r = paired_test(A[col].to_numpy(), Bm[col].to_numpy())
            tests.append(dict(comparison=f"Proposed Pareto vs {METHOD_NAMES[other]}", metric=lab, **r))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = holm(tests["p"].to_numpy())
    tests.to_csv(out("table_stat_tests.csv"), index=False)

    # =============================================== main-seed artefacts: Pareto figure, biology
    R = results[seeds[0]]
    g, cands, sel, F = R["g"], R["cands"], R["sel"], R["F"]
    front = sel["front"]
    pf = pd.DataFrame(F[front], columns=["paths", "worst_cost", "uncertainty_f3", "neg_confidence"])
    pf["confidence"] = -pf.pop("neg_confidence")
    pf.sort_values("paths").to_csv(out("pareto_front_main_seed.csv"), index=False)
    pd.DataFrame(F, columns=["paths", "worst_cost", "uncertainty_f3", "neg_confidence"]).to_csv(out("candidates_main_seed.csv"), index=False)
    marks = {"Balanced (proposed)": sel["idx"]["proposed"], "Min-cardinality": sel["idx"]["min_card"],
             "Min worst-cost": sel["idx"]["min_wc"]}
    for pr in R["prefs"]:
        if pr["preference"] != "Balanced":
            marks[pr["preference"]] = pr["cand"]
    report.fig_pareto(F, front, marks, out("fig1_pareto_front.png"), f"(top-{cfg.main_size}, seed {seeds[0]}, {len(front)} solutions)")

    bio_rows, cover_hits = [], None
    if not cfg.skip_reactome and not cfg.synthetic:
        print("\n=== Reactome enrichment on the predefined balanced solution (and baselines) ===")
        for m in ("proposed", "min_card", "min_wc"):
            summ, hits = bio_analysis(g, cands, sel["idx"][m], m, cfg, cfg.outdir)
            bio_rows.append(summ)
            if m == "proposed":
                cover_hits = hits
        pd.DataFrame(bio_rows).to_csv(out("table_reactome_comparison.csv"), index=False)
    cov = cands.covers[sel["idx"]["proposed"]]
    report.fig_biology(g, cands, cov, path_conf_np(cands.pool, g.mu_e), cover_hits, out("fig2_biology.png"))

    # =============================================== summary
    with open(out("RESULTS_SUMMARY.md"), "w") as f:
        f.write(f"# Results summary (top-{cfg.main_size} network, {len(seeds)} seeds, B={cfg.mc_B})\n\n")
        f.write("Objectives frozen: f1=k, f2=max_r R(L(P_r)), f3=0.7*mean u+0.3*Q90(u), f4=-mean confidence.\n\n")
        f.write("## Table 1 - main comparison (mean ± SD over seeds)\n" + md(t1) + "\n\n")
        f.write("## Table 2 - preference sensitivity\n" + md(t2) + "\n\n")
        f.write("## Table 3 - robustness (secondary analysis)\n" + md(t3) + "\n\n")
        f.write("## Table 4 - scalability (target selection vs actual size)\n" + md(scal) + "\n\n")
        f.write("Candidate generation is capped by the stated budget; covers were NOT exhaustively enumerated.\n\n")
        f.write("## Paired tests (Wilcoxon signed-rank, Holm-adjusted)\n" + md(tests) + "\n\n")
        if bio_rows:
            f.write("## Reactome comparison\n" + md(pd.DataFrame(bio_rows)) + "\n")
    print("\n", t1.to_string(index=False))
    print("\n", t2.to_string(index=False))
    print("\n", t3.to_string(index=False))
    print(f"\nDone. Everything is in ./{cfg.outdir}/ (start with RESULTS_SUMMARY.md)")


if __name__ == "__main__":
    main()
