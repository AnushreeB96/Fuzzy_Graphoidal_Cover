# Multi-objective fuzzy graphoidal cover on TCGA-BRCA (GPU / CUDA) - final

Reproducible experiment for the manuscript: fuzzy graphoidal cover (Algorithm 3.3) as a multi-objective
problem, evaluated on BRCA signalling sub-networks (cBioPortal TCGA PanCancer Atlas + STRING + Reactome).

**Frozen objectives (do not change between runs):**

| | objective (minimise) | definition |
|---|---|---|
| f1 | number of paths | k |
| f2 | worst-path fuzzy cost | max_r R(L(P_r)) |
| f3 | uncertainty | 0.7·mean_r u_r + 0.3·Q0.90(u), u_r = (L^U − L^L)/(L^M + ε) |
| f4 | negative mean confidence | −(1/k) Σ_r s_r, s_r = geometric mean of μ_E on path r |

Methods (all select from the **same** candidate set): Minimum-cardinality · Minimum worst-cost ·
**Proposed Pareto** (balanced weights) · **Robustness-aware Pareto** (secondary analysis).

---

## 1. Quick start in Google Colab (step by step)

**Fastest way:** upload `fgc_brca.zip` **and** open `run_in_colab.ipynb` in Colab (File → Upload notebook; it is also inside the zip).
It contains every cell below, ready to run top to bottom. The manual steps follow.

### Step 0 - open a GPU notebook
1. Go to <https://colab.research.google.com> → **New notebook**.
2. Menu **Runtime → Change runtime type → Hardware accelerator: GPU (T4)** → Save.
3. Run this cell to confirm a GPU is attached:
```python
!nvidia-smi
import torch; print(torch.__version__, "CUDA available:", torch.cuda.is_available())
```
(If it prints `False`, the runtime is not GPU - fix Step 0.2. The code still runs on CPU but much slower.)

### Step 1 - get the code into Colab
Pick **one** option.

**Option A - upload the zip (simplest).** Left sidebar → folder icon → upload `fgc_brca.zip` to `/content`, then:
```python
!unzip -q -o /content/fgc_brca.zip -d /content
%cd /content/fgc_brca
!ls
```
You should see `run_brca.py  tests_numpy.py  requirements.txt  README.md  fgc/`.

**Option B - via Google Drive (keeps results after the session ends).**
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q -o "/content/drive/MyDrive/fgc_brca.zip" -d /content     # zip placed in My Drive
%cd /content/fgc_brca
```
Later, add `--outdir /content/drive/MyDrive/fgc_results` and `--data_dir /content/drive/MyDrive/fgc_data`
to the run command so results **and downloaded data** survive a disconnect.

### Step 2 - install the (few) extra packages
```python
!pip -q install -r requirements.txt
```
PyTorch is already installed on Colab; do not reinstall it.

### Step 3 - sanity checks (about 1 minute)
```python
!python tests_numpy.py                       # CPU test of the combinatorial core -> prints ALL OK
!python run_brca.py --synthetic --quick      # offline smoke test of the full GPU pipeline (no downloads)
```
If the smoke test finishes with `Done. Everything is in ./results/` you are ready. If it crashes, copy the
**last 30 lines of the traceback** and send them to me - the GPU path has not been executed outside Colab.

### Step 4 - the real BRCA experiment
Do a short trial run first (recommended), then the full one:
```python
# 4a. trial (about 10-30 min; downloads data on first run ~ 1-2 GB for STRING + TCGA files)
!python run_brca.py --n_seeds 2 --mc_B 20 --robust_B 20

# 4b. FULL run for the paper (10 seeds, B = 100 perturbations; can take hours - see Section 3)
!python run_brca.py --outdir results_final
```
Keep the browser tab open (or use Colab Pro); free Colab disconnects after ~12 h or when idle.
**Checkpoints:** every finished seed is saved to `<outdir>/checkpoints/`. If the session dies, reconnect, redo Steps 1-2,
and **rerun the exact same command**: finished seeds are skipped (`--no_resume` forces a recompute; a checkpoint made with
different settings is ignored automatically). Use Drive paths (`--outdir /content/drive/MyDrive/fgc_results
--data_dir /content/drive/MyDrive/fgc_data`) so checkpoints and downloaded data survive.

### Step 5 - collect the results
```python
!cat results_final/RESULTS_SUMMARY.md                 # all tables in one file
!cd results_final && zip -qr ../results_final.zip . && ls -la ../results_final.zip
from google.colab import files; files.download('/content/fgc_brca/results_final.zip')
```
Figures: `results_final/fig1_pareto_front.png` and `fig2_biology.png` (also `.pdf`). View inline:
```python
from IPython.display import Image, display
display(Image('results_final/fig1_pareto_front.png')); display(Image('results_final/fig2_biology.png'))
```

---

## 2. What each output file is

| File (in `--outdir`, default `results/`) | Manuscript item |
|---|---|
| `RESULTS_SUMMARY.md` | all tables below, ready to read |
| `table1_main_comparison.csv` | **Table 1**: 4 methods × paths, worst cost, confidence, uncertainty, runtime, ARI ±5/10/20% (mean ± SD over seeds) |
| `table2_preference_sensitivity.csv`, `table2_preference_main_seed.csv` | **Table 2**: compactness / balanced / uncertainty-oriented (weights fixed in `fgc/config.py`) |
| `table3_robustness.csv` | **Table 3** (secondary): ARI, tie-aware ARI, Δk, cost change, #equally-optimal covers |
| `table4_scalability.csv` | **Table 4**: *target selection* (Top-50/100/200/500) vs **actual** genes, edges, candidate budget/distinct candidates, "budget cap reached", Pareto size, runtime, memory |
| `table_stat_tests.csv` | paired Wilcoxon (Proposed vs each other method), rank-biserial r, Cohen's dz, Holm-adjusted p |
| `fig1_pareto_front.png/.pdf` | **Figure 1**: paths vs worst cost; colour = confidence (left), uncertainty (right) |
| `fig2_biology.png/.pdf` | **Figure 2**: selected balanced cover network + Reactome enrichment |
| `table_reactome_paths_{proposed,min_card,min_wc}.csv` | per-path: #genes, confidence, fuzzy cost/length, uncertainty, top Reactome pathway, FDR |
| `table_reactome_comparison.csv`, `reactome_cover_*.csv` | proposed vs the two baselines: how many paths/pathways are significant |
| `seed_results.csv`, `seed_preferences.csv` | raw per-seed numbers behind Tables 1-3 |
| `robustness_realizations.csv` | **every perturbation**: selected cover ID, number of equally-optimal covers, ARI, tie-aware ARI |
| `robustness_full_by_seed.csv`, `pareto_front_main_seed.csv`, `candidates_main_seed.csv` | supporting data |
| `data_info.json`, `config.json` | STRING version/threshold/organism, exact settings (cite in Methods) |

---

## 3. Run time, memory and useful options

Cost is dominated by the robustness experiment: for each seed, (3 levels × `mc_B` + 1) realisations, each with a nested
`robust_B × 3`-run ensemble. Rough guidance (T4 GPU; **estimates, not measured**):

| Goal | Command | Expect |
|---|---|---|
| Smoke test | `--synthetic --quick` | ~1-3 min |
| Trial on real data | `--n_seeds 2 --mc_B 20 --robust_B 20` | tens of minutes |
| Final paper run | *(defaults: 10 seeds, `--mc_B 100 --robust_B 100`)* | hours |
| Cheaper final run | `--mc_B 50 --robust_B 50` | roughly ¼ of the above |

A long run needs no manual chunking: just rerun the same command after a disconnect (see checkpoints, Step 4).

Options (`python run_brca.py -h` lists all):
`--n_seeds 10` · `--seed 42` (first seed) · `--main_size 200` · `--sizes 50 100 200 500` · `--mc_B 100` · `--robust_B 100` ·
`--robust_deltas 0.05 0.10 0.20` · `--robust_kappa 0.5` · `--n_candidates N` (overrides budgets 500/1000/1000/5000) ·
`--lam a b c d` · `--no_resume` · `--skip_reactome` · `--outdir DIR` · `--data_dir DIR` · `--device cuda|cpu` · `--quick`.

**Out of GPU memory** (mostly on the 500-gene setting): `--n_candidates 2000`, or `--sizes 50 100 200`.

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPU found - CPU` printed | Runtime → Change runtime type → GPU, reconnect, rerun from Step 1 |
| Download fails / 404 for cBioPortal or STRING | URLs can change. Download manually and place: `data/brca_tcga_pan_can_atlas_2018/` (needs `data_mrna_seq_v2_rsem.txt`, `data_mutations.txt`, `data_cna.txt`, and one `..._zscores_ref_*.txt`) and `data/string/links.txt.gz`, `data/string/info.txt.gz` (STRING v12.0 human protein.links and protein.info). Then rerun. |
| Reactome step prints `n/a` | Reactome API unreachable/rate-limited; rerun later or use `--skip_reactome` |
| `CUDA out of memory` | see Section 3 |
| Session disconnected mid-run | reconnect, redo Steps 1-2, rerun the **same** command (finished seeds are loaded from checkpoints; data is cached in `--data_dir`) |
| `can't open file '/content/run_brca.py'` | wrong folder: run `%cd /content/fgc_brca` first |
| Old error keeps appearing after re-uploading the zip | Colab kept the old copy: `!rm -rf /content/fgc_brca /content/fgc_brca*.zip`, upload again, unzip again |
| Any Python traceback | send me the last ~30 lines |

---

## 5. Method notes to state in the paper (honest reporting)

* **Candidates are heuristic, not exhaustive.** Randomised greedy assemblies from fuzzy shortest paths; budget 500/1000/1000/5000 for Top-50/100/200/500.
  Table 4 reports the distinct number obtained and whether the cap was reached. "Minimum-cardinality" = best cover *within the candidate set*.
* **Target vs actual size.** "Top-N" is the number of genes ranked by vertex membership; isolated genes (no STRING edge ≥ 0.70 inside the subset) are dropped, so actual gene counts are smaller.
* **Tie handling.** Minimum-cardinality breaks ties lexicographically (f1, f2, f3, f4); minimum worst-cost by (f2, f1, f3, f4). Because k does not change under perturbation,
  a unique minimum-k cover is *exactly* stable (ARI = 1): check `n_equally_optimal_nominal` in Table 3 and `robustness_realizations.csv` to see whether ties exist;
  the tie-aware ARI is the best agreement with any equally-optimal nominal cover.
* **Perturbation** acts on memberships (μ' = clip(μ+U(−δ,δ),0,1)) and on fuzzy lengths (L' = L·(1+U(−δ,δ)) per component, then re-sorted). All methods re-select from the same fixed candidate set.
* **Robustness-aware Pareto** is a secondary analysis: objectives f^rob = mean + 0.5·SD over 100 runs at each of ±5/10/20% (fixed a priori). It is **not guaranteed** to be more stable than the ordinary Pareto pick - report whatever you find.
* **Seeds** change the bootstrap edge lengths, candidate generation and perturbation draws. Tables 1-3 report mean ± SD across seeds; Fig. 1 and the Reactome analysis use the first seed and the *predefined balanced* weights (0.25 each), for the proposed cover **and** both baselines.
* **Statistics:** paired two-sided Wilcoxon signed-rank across seeds (min. attainable p with 10 seeds ≈ 0.002), rank-biserial correlation and Cohen's dz as effect sizes, Holm correction over all comparisons.
* Carried-over modelling assumptions (edit `fgc/tfn.py` if your Section 2.3 differs): ranking R = (l+2m+u)/4; totally governing = all three points ≤; fractionally governing = ≥ 2 points ≤ and R(A) ≤ R(B); Step 4 replaced by greedy conflict resolution that enforces the graphoidal definition (verified for every selected cover); μ_E = c·√(μ_V(i)μ_V(j)) can exceed min(μ_V) (`--clip_edge_membership`).
* Sparse-tensor invariant checks are enabled explicitly and the framework notice is silenced, so logs stay clean.

## 6. Files
```
run_brca.py        main script (scalability, multi-seed experiment, tables, figures, Reactome)
tests_numpy.py     CPU sanity test          requirements.txt     run_in_colab.ipynb  ready-made Colab notebook
fgc/config.py      settings + preference scenarios   fgc/tfn.py  fgc/dijkstra.py  fgc/cover.py   (Algorithm 3.3, candidates)
fgc/objectives.py  GPU objectives, Pareto, balanced selection, ARI   fgc/pipeline.py  fgc/robustness.py
fgc/graph.py       GPU bootstrap → fuzzy lengths     fgc/data.py  cBioPortal + STRING     fgc/reactome.py
fgc/stats.py       paired tests      fgc/report.py  figures     fgc/synthetic.py  offline test data
```

**Verification status:** `tests_numpy.py` and the statistics helpers pass in my sandbox. The PyTorch/GPU code and the downloads could not be executed there
(no torch, no internet), so Step 3 is the real first test - send me any traceback.
