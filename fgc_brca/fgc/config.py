from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _cand_default() -> Dict[int, int]:
    # number of randomised greedy assemblies (upper bound on distinct candidate covers) per network size
    return {50: 500, 100: 1000, 200: 1000, 500: 5000}


@dataclass
class Config:
    # ---- paths / runtime
    outdir: str = "results"
    data_dir: str = "data"
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    seed: int = 42
    synthetic: bool = False       # offline smoke-test data instead of TCGA/STRING
    skip_reactome: bool = False

    # ---- data (Sec. 7.1-7.3)
    string_threshold: float = 0.70
    alpha: float = 0.5
    n_boot: int = 200
    clip_edge_membership: bool = False

    # ---- networks (Sec. 7.4)
    main_size: int = 200
    sizes: Tuple[int, ...] = (50, 100, 200, 500)

    # ---- Algorithm 3.3
    n_sources: int = 25
    max_labels: int = 4
    strength_mode: str = "none"   # "none" | "inverse"
    rank_mode: str = "yager"      # TFN ranking R(.)

    # ---- candidate covers (identical set is shared by ALL methods)
    n_candidates: Optional[int] = None            # override; None -> cand_by_size
    cand_by_size: Dict[int, int] = field(default_factory=_cand_default)
    eps: float = 1e-9

    # ---- objectives: f = (k, max_r c_r, 0.7*mean u + 0.3*Q90(u), -mean_r s_r); compromise weights in this order
    lam: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    # ---- robust proposed method: f^R = E[f] + kappa*SD[f]; memberships AND fuzzy lengths are perturbed
    robust_deltas: Tuple[float, ...] = (0.05, 0.10, 0.20)   # ensemble = robust_B runs per level, pooled
    robust_B: int = 100
    robust_kappa: float = 0.5     # predefined risk aversion (gamma in the write-up)

    # ---- robustness experiment (outer Monte-Carlo)
    deltas: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    mc_B: int = 100
    stability_delta: float = 0.10

    # ---- reproducibility / biology
    n_seeds: int = 10             # independent seeds (bootstrap, candidates, perturbations) for the main network
    reactome_max_paths: int = 60  # max multi-gene paths enriched per cover
    resume: bool = True           # reuse finished per-seed checkpoints found in <outdir>/checkpoints


# Decision-maker preference scenarios, fixed BEFORE looking at results.
# Order: (paths, worst-path cost, worst-path uncertainty, -confidence)
PREFERENCES = {
    "Compactness-oriented": (0.50, 0.20, 0.15, 0.15),
    "Balanced": (0.25, 0.25, 0.25, 0.25),
    "Uncertainty-oriented": (0.15, 0.20, 0.45, 0.20),
}
