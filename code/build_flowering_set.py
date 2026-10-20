# -*- coding: utf-8 -*-
"""
build_flowering_set.py — literature flowering-time gene set, independent of AtMAD.

Gene symbols are canonical flowering-time regulators of the photoperiod,
vernalization, autonomous, ageing, gibberellin and floral-integrator
pathways (reviewed by Andres & Coupland, Nat. Rev. Genet. 2012). Each symbol
is mapped to its TAIR locus through reviewed UniProtKB entries of
Arabidopsis thaliana (organism 3702); symbols whose reviewed entries map to
more than one locus, or to none, are discarded.

Output: data/flowering_literature.csv (symbol, locus, uniprot, protein).
"""
import os, sys, time, csv, io
import urllib.request, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

SYMBOLS = [
    # floral integrators / meristem identity
    "FT", "TSF", "SOC1", "AGL24", "FD", "LFY", "AP1", "FUL", "TFL1", "SVP", "FLC", "FLM",
    "MAF2", "MAF4", "MAF5",
    # FLC activators / vernalization
    "FRI", "FES1", "SUF4", "FLX", "PIE1", "EFS", "ELF7", "ELF8", "VIN3", "VRN1", "VRN2",
    # autonomous pathway
    "FCA", "FPA", "FVE", "FLD", "LD", "FY", "FLK",
    # photoperiod and circadian clock
    "CO", "GI", "FKF1", "CDF1", "ZTL", "ELF3", "ELF4", "LUX", "LHY", "CCA1", "TOC1",
    "PRR5", "PRR7", "PRR9", "PHYA", "PHYB", "CRY1", "CRY2", "COP1", "SPA1", "HOS1", "PFT1",
    # ageing pathway and repressors of FT
    "SPL3", "SPL9", "SPL15", "TOE1", "TOE2", "SMZ", "SNZ", "TEM1", "TEM2",
    # gibberellin
    "GAI", "RGA", "SPY",
]
URL = "https://rest.uniprot.org/uniprotkb/search?"


def query(sym):
    q = f"(gene_exact:{sym}) AND (organism_id:3702) AND (reviewed:true)"
    url = URL + urllib.parse.urlencode({"query": q, "fields": "accession,gene_primary,gene_names,xref_tair,protein_name",
                                        "format": "tsv", "size": 25})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))
        except Exception:
            time.sleep(2 + 2 * attempt)
    return None


def main():
    rows, dropped = [], []
    for s in SYMBOLS:
        res = query(s)
        if res is None:
            dropped.append((s, "request failed")); continue
        hits = [r for r in res if r["Gene Names (primary)"].strip().upper() == s]
        if not hits:   # symbol is a synonym of the UniProt primary name (e.g. FUL = AGL8)
            hits = [r for r in res if s in r["Gene Names"].upper().split()]
        loci = sorted({t.strip().upper() for r in hits for t in r["TAIR"].split(";") if t.strip()})
        if len(loci) != 1:
            dropped.append((s, f"{len(loci)} loci")); continue
        h = hits[0]
        rows.append({"symbol": s, "locus": loci[0], "uniprot_primary": h["Gene Names (primary)"], "uniprot": h["Entry"], "protein": h["Protein names"][:80]})
        time.sleep(0.3)
    out = os.path.join(C.PROJECT_DIR, "data", "flowering_literature.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "locus", "uniprot_primary", "uniprot", "protein"])
        w.writeheader(); w.writerows(rows)
    print(f"kept {len(rows)} / {len(SYMBOLS)} symbols -> {out}")
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
