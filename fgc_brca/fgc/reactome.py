"""Reactome over-representation analysis via the public Analysis Service REST API.
Run AFTER the cover is fixed; never used to build it.  FDR = Benjamini-Hochberg adjusted p-value from Reactome."""
from __future__ import annotations
import time
import requests

URL = ("https://reactome.org/AnalysisService/identifiers/projection"
       "?pageSize=15&page=1&sortBy=ENTITIES_FDR&order=ASC&resource=TOTAL&pValue=1&includeDisease=true")


def enrich(genes, retries=2, timeout=60, pause=0.2):
    """Returns a list of dicts (name, stId, pvalue, fdr, found, total) sorted by FDR, or [] on failure."""
    body = "#genes\n" + "\n".join(genes)
    for _ in range(retries + 1):
        try:
            r = requests.post(URL, data=body, headers={"Content-Type": "text/plain"}, timeout=timeout)
            r.raise_for_status()
            out = []
            for p in r.json().get("pathways", []):
                en = p.get("entities", {})
                out.append(dict(name=p.get("name", "?"), stId=p.get("stId", ""), pvalue=en.get("pValue"),
                                fdr=en.get("fdr"), found=en.get("found"), total=en.get("total")))
            time.sleep(pause)
            return out
        except Exception:
            time.sleep(2)
    return []
