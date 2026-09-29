"""Data acquisition (cBioPortal TCGA-BRCA PanCancer Atlas + STRING) and BRCA fuzzy memberships (Sec. 7.1-7.2)."""
from __future__ import annotations
import glob
import gzip
import os
import tarfile
from dataclasses import dataclass
import numpy as np
import pandas as pd

STUDY = "brca_tcga_pan_can_atlas_2018"
STUDY_URLS = [
    f"https://datahub.assets.cbioportal.org/{STUDY}.tar.gz",
    f"https://cbioportal-datahub.s3.amazonaws.com/{STUDY}.tar.gz",
]
STRING_VERSION = "v12.0"
STRING_LINKS = [
    f"https://stringdb-downloads.org/download/protein.links.{STRING_VERSION}/9606.protein.links.{STRING_VERSION}.txt.gz",
    f"https://stringdb-static.org/download/protein.links.{STRING_VERSION}/9606.protein.links.{STRING_VERSION}.txt.gz",
]
STRING_INFO = [
    f"https://stringdb-downloads.org/download/protein.info.{STRING_VERSION}/9606.protein.info.{STRING_VERSION}.txt.gz",
    f"https://stringdb-static.org/download/protein.info.{STRING_VERSION}/9606.protein.info.{STRING_VERSION}.txt.gz",
]
NONCODING = ("Silent", "Intron", "IGR", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "RNA", "Splice_Region")


@dataclass
class BRCAData:
    genes: list
    expr: np.ndarray          # (G,n) log2(RSEM+1)
    A: np.ndarray             # normalised alteration score in [0,1]
    E: np.ndarray             # normalised expression signal in [0,1]
    mu_v: np.ndarray
    string_edges: pd.DataFrame  # columns a,b,score (a<b, score in [0,1])
    info: dict


def _download(urls, dest):
    import requests
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for u in urls:
        try:
            print(f"[download] {u}")
            with requests.get(u, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(dest + ".part", dest)
            return dest
        except Exception as ex:  # try next mirror
            last = ex
            print(f"   failed: {ex}")
    raise RuntimeError(f"All mirrors failed for {dest}: {last}\n"
                       f"Download manually and place the file at {dest}")


def _find(folder, patterns):
    for p in patterns:
        hits = glob.glob(os.path.join(folder, "**", p), recursive=True)
        if hits:
            return hits[0]
    return None


def _read_matrix(path):
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df = df[df["Hugo_Symbol"].notna()]
    df = df.drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
    if "Entrez_Gene_Id" in df.columns:
        df = df.drop(columns=["Entrez_Gene_Id"])
    return df.apply(pd.to_numeric, errors="coerce")


def load_string(cfg):
    d = os.path.join(cfg.data_dir, "string")
    cache = os.path.join(d, f"edges_{STRING_VERSION}_{int(cfg.string_threshold * 1000)}.tsv.gz")
    if os.path.exists(cache):
        return pd.read_csv(cache, sep="\t")
    lk = _download(STRING_LINKS, os.path.join(d, "links.txt.gz"))
    inf = _download(STRING_INFO, os.path.join(d, "info.txt.gz"))
    info = pd.read_csv(inf, sep="\t", usecols=[0, 1])
    info.columns = ["pid", "name"]
    pid2name = dict(zip(info.pid, info.name))
    links = pd.read_csv(lk, sep=" ")
    links = links[links["combined_score"] >= int(round(cfg.string_threshold * 1000))]
    links["a"] = links["protein1"].map(pid2name)
    links["b"] = links["protein2"].map(pid2name)
    links = links.dropna(subset=["a", "b"])
    links = links[links.a != links.b]
    lo = np.where(links.a < links.b, links.a, links.b)
    hi = np.where(links.a < links.b, links.b, links.a)
    out = pd.DataFrame({"a": lo, "b": hi, "score": links["combined_score"].to_numpy() / 1000.0})
    out = out.groupby(["a", "b"], as_index=False)["score"].max()
    out.to_csv(cache, sep="\t", index=False)
    return out


def load_brca(cfg):
    folder = os.path.join(cfg.data_dir, STUDY)
    if not os.path.isdir(folder) or _find(folder, ["data_mrna_seq_v2_rsem.txt"]) is None:
        tgz = _download(STUDY_URLS, os.path.join(cfg.data_dir, f"{STUDY}.tar.gz"))
        with tarfile.open(tgz) as t:
            t.extractall(cfg.data_dir)
    f_expr = _find(folder, ["data_mrna_seq_v2_rsem.txt"])
    f_z = _find(folder, ["data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
                         "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"])
    f_mut = _find(folder, ["data_mutations.txt", "data_mutations_extended.txt"])
    f_cna = _find(folder, ["data_cna.txt", "data_CNA.txt"])
    assert f_expr and f_mut and f_cna, f"missing study files in {folder}"

    print("[data] reading expression ...")
    rsem = _read_matrix(f_expr)
    samples = list(rsem.columns)
    logx = np.log2(rsem.clip(lower=0).fillna(0) + 1.0)
    logx = logx[(logx.var(axis=1) > 0)]

    # ---- z-score signal E_i = min(|median z_i| / zmax, 1)
    if f_z:
        z = _read_matrix(f_z)
        z = z[[c for c in z.columns if c in set(samples)]]
        zsrc = os.path.basename(f_z)
    else:
        z = (logx.sub(logx.mean(1), axis=0)).div(logx.std(1) + 1e-9, axis=0)
        zsrc = "computed z-score over all BRCA samples (median ~ 0; prefer the cBioPortal file)"
    medz = z.median(axis=1).abs()

    # ---- alteration frequency A_i (non-silent somatic mutation OR GISTIC |CNA|==2)
    print("[data] reading mutations / CNA ...")
    mut = pd.read_csv(f_mut, sep="\t", comment="#", low_memory=False,
                      usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"])
    mut = mut[~mut["Variant_Classification"].isin(NONCODING)]
    mut = mut[mut["Tumor_Sample_Barcode"].isin(set(samples))]
    altered = {}
    for g, s in zip(mut["Hugo_Symbol"], mut["Tumor_Sample_Barcode"]):
        altered.setdefault(g, set()).add(s)
    cna = _read_matrix(f_cna)
    cna = cna[[c for c in cna.columns if c in set(samples)]]
    hi = cna.abs().eq(2)
    for g in hi.index[hi.any(axis=1)]:
        altered.setdefault(g, set()).update(hi.columns[hi.loc[g].to_numpy()])
    nS = len(samples)

    print("[data] reading STRING ...")
    ed = load_string(cfg)
    in_net = set(ed.a) | set(ed.b)
    genes = [g for g in logx.index if g in in_net]
    A = np.array([len(altered.get(g, ())) / nS for g in genes])
    A = A / (A.max() + 1e-12)
    zz = medz.reindex(genes).fillna(0.0).to_numpy()
    E = np.minimum(zz / (zz.max() + 1e-12), 1.0)
    mu_v = cfg.alpha * A + (1 - cfg.alpha) * E
    gset = set(genes)
    ed = ed[ed.a.isin(gset) & ed.b.isin(gset)].reset_index(drop=True)
    info = dict(study=STUDY, n_samples=nS, n_genes_expression=int(len(logx)),
                n_genes_in_string_network=len(genes), n_string_edges=len(ed),
                string_version=STRING_VERSION, string_threshold=cfg.string_threshold,
                organism="Homo sapiens (9606)", network_type="full STRING network, combined score",
                zscore_source=zsrc, alpha=cfg.alpha)
    print("[data]", info)
    return BRCAData(genes, logx.loc[genes].to_numpy(dtype=np.float32), A, E, mu_v, ed, info)
