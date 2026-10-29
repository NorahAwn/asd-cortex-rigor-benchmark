#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
donor_overlap_audit.py  --  Shared-control-donor audit across public ASD cohorts (TCBB)
========================================================================================
Demonstrates, on REAL metadata, the shared-control confound the review warns about:
cohorts produced by related groups often reuse the same neurotypical control brains,
which inflates cross-cohort concordance independently of disease.

It finds, among CONTROL donors only:
  (A) exact donor-ID matches across cohorts
  (B) metadata matches (same sex, age/PMI/RIN within tolerance) -- probable shared brains
and writes a table of the flagged overlaps.

Optionally (--sigA/--sigB), it reports the cross-cohort signature correlation
("with shared controls"); the "without" value comes from re-running benchmark_de.py
on each cohort after excluding the flagged donors (the script prints the exclusion list).

------------------------------------------------------------------------------------
INPUT
  One metadata table per cohort (CSV/TSV), one row per donor, with columns for:
  donor id, diagnosis, and any of: sex, age, pmi, rin. Column names are configurable.

USAGE
  pip install pandas numpy scipy
  python3 donor_overlap_audit.py \
      --cohort Velmeshev2019=velmeshev_donors.csv \
      --cohort Wamsley2024=wamsley_donors.csv \
      --cohort Perez2025=perez_donors.csv \
      --id-col donor_id --dx-col diagnosis --control Control \
      --sex-col sex --age-col age --pmi-col pmi --rin-col rin \
      --age-tol 2 --pmi-tol 3 --rin-tol 0.6 --out results

  # optional concordance ("with shared controls") between two cohorts:
  #   --sigA velmeshev_L23_logfc.csv --sigB wamsley_L23_logfc.csv --sig-gene gene --sig-val log2FoldChange
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd

def read_any(path):
    sep="\t" if path.lower().endswith((".tsv",".txt")) else ","
    return pd.read_csv(path, sep=sep)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", required=True,
                    help="NAME=path.csv  (repeat for each cohort)")
    ap.add_argument("--id-col", default="donor_id")
    ap.add_argument("--dx-col", default="diagnosis")
    ap.add_argument("--control", default="Control")
    ap.add_argument("--sex-col", default="sex")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--pmi-col", default="pmi")
    ap.add_argument("--rin-col", default="rin")
    ap.add_argument("--age-tol", type=float, default=2.0)
    ap.add_argument("--pmi-tol", type=float, default=3.0)
    ap.add_argument("--rin-tol", type=float, default=0.6)
    ap.add_argument("--sigA", default=None); ap.add_argument("--sigB", default=None)
    ap.add_argument("--sig-gene", default="gene"); ap.add_argument("--sig-val", default="log2FoldChange")
    ap.add_argument("--out", default="results")
    a=ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cohorts={}
    for spec in a.cohort:
        if "=" not in spec: sys.exit(f"[error] --cohort must be NAME=path, got '{spec}'")
        name,path=spec.split("=",1)
        df=read_any(path)
        if a.id_col not in df.columns or a.dx_col not in df.columns:
            sys.exit(f"[error] {name}: need columns '{a.id_col}' and '{a.dx_col}'. Have {list(df.columns)}")
        df=df[df[a.dx_col].astype(str).str.lower().str.startswith(a.control.lower())].copy()
        df["_cohort"]=name; df["_id"]=df[a.id_col].astype(str).str.strip()
        cohorts[name]=df
        print(f"[load] {name}: {len(df)} control donors")

    names=list(cohorts); exact=[]; fuzzy=[]
    for n1,n2 in itertools.combinations(names,2):
        d1,d2=cohorts[n1],cohorts[n2]
        # (A) exact id
        shared=set(d1["_id"])&set(d2["_id"])
        for s in sorted(shared): exact.append(dict(cohort_A=n1,cohort_B=n2,donor_A=s,donor_B=s,match="exact-ID"))
        # (B) metadata match
        def has(c): return c in d1.columns and c in d2.columns
        for _,r1 in d1.iterrows():
            for _,r2 in d2.iterrows():
                if r1["_id"]==r2["_id"]: continue
                ok=True; reasons=[]
                if has(a.sex_col):
                    if str(r1[a.sex_col]).lower()[:1]!=str(r2[a.sex_col]).lower()[:1]: ok=False
                    else: reasons.append("sex")
                for col,tol,nm in [(a.age_col,a.age_tol,"age"),(a.pmi_col,a.pmi_tol,"PMI"),(a.rin_col,a.rin_tol,"RIN")]:
                    if has(col):
                        try:
                            if abs(float(r1[col])-float(r2[col]))<=tol: reasons.append(nm)
                            else: ok=False
                        except (ValueError,TypeError): pass
                if ok and len(reasons)>=2:
                    fuzzy.append(dict(cohort_A=n1,cohort_B=n2,donor_A=r1["_id"],donor_B=r2["_id"],
                                      match="meta:"+"+".join(reasons)))

    ex=pd.DataFrame(exact); fz=pd.DataFrame(fuzzy)
    allov=pd.concat([ex,fz],ignore_index=True) if len(exact)+len(fuzzy) else pd.DataFrame(
        columns=["cohort_A","cohort_B","donor_A","donor_B","match"])
    csv=os.path.join(a.out,"donor_overlap.csv"); allov.to_csv(csv,index=False)

    # exclusion list per cohort (donors flagged as shared)
    excl={}
    for _,row in allov.iterrows():
        excl.setdefault(row["cohort_A"],set()).add(row["donor_A"])
        excl.setdefault(row["cohort_B"],set()).add(row["donor_B"])

    print("\n================= RESULTS TO PASTE BACK =================")
    print(f"cohorts audited: {', '.join(names)}")
    print(f"exact donor-ID control overlaps ...... {len(ex)}")
    print(f"metadata-matched control overlaps .... {len(fz)}  (>=2 fields within tolerance)")
    if len(allov):
        print("\nflagged overlaps:")
        print(allov.to_string(index=False))
    for n in names:
        s=sorted(excl.get(n,set()))
        print(f"\n[{n}] exclude these {len(s)} control donors before cross-cohort concordance:")
        print("  "+(", ".join(s) if s else "(none)"))

    # optional: cross-cohort signature concordance ("with shared controls")
    if a.sigA and a.sigB:
        from scipy.stats import pearsonr, spearmanr
        A=read_any(a.sigA); B=read_any(a.sigB)
        A=A.set_index(a.sig_gene)[a.sig_val]; B=B.set_index(a.sig_gene)[a.sig_val]
        g=A.index.intersection(B.index)
        x=A.loc[g].values; y=B.loc[g].values
        m=np.isfinite(x)&np.isfinite(y)
        print(f"\ncross-cohort signature concordance WITH shared controls (n={m.sum()} genes):")
        print(f"  Pearson r  = {pearsonr(x[m],y[m])[0]:.3f}")
        print(f"  Spearman r = {spearmanr(x[m],y[m])[0]:.3f}")
        print("  -> re-run benchmark_de.py on each cohort EXCLUDING the donors listed above,")
        print("     regenerate the two signature files, and re-run with --sigA/--sigB to get the WITHOUT value.")
    print("CSV written to:", csv)
    print("========================================================\n")

if __name__=="__main__":
    main()
