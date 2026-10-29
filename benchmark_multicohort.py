#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_multicohort.py
========================
Two-cohort donor-level benchmark for the TCBB ASD single-cell review.

For BOTH open cohorts (Velmeshev 2019 and Perez/dup15q 2025), using the
pre-built donor x region x cell-type pseudobulk matrices, this script:

  * aggregates to donor x cell-type (donor is the unit of replication);
  * runs TWO donor-level estimators on IDENTICAL donor sets / permutations:
        - raw-count DESeq2 pseudobulk      (negative-binomial count model)
        - linear-model pseudobulk          (OLS on log2 CPM; limma/voom family)
  * reports the REAL case-vs-control DEG count (unadjusted and covariate-
    adjusted) per cell type;
  * runs a randomized-label negative control over N identical permutations
    and reports, per estimator, the mean/SD/max DEG count AND the
    REJECTION FREQUENCY (fraction of permutations yielding >=1 rejection);
  * runs split-half replication (Spearman of donor-level linear-model
    t-statistics across disjoint donor halves);
  * prints a cross-cohort comparison so the calibration pattern can be seen
    to replicate across two independent datasets.

Outputs (in --out):
  benchmark_multicohort_summary.csv   -> the numbers for the manuscript table
  benchmark_multicohort_perperm.csv   -> per-permutation DEG counts (inspect)

Everything is deterministic given --seed. No R required.

Typical run (Windows PowerShell, with .venv active):
  python benchmark_multicohort.py --nc-reps 30 --split-reps 10
Faster first pass to gauge runtime:
  python benchmark_multicohort.py --nc-reps 10 --split-reps 5
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ---- DESeq2 (PyDESeq2) ------------------------------------------------------
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    HAVE_DESEQ = True
except Exception as e:
    HAVE_DESEQ = False
    _DESEQ_ERR = str(e)

NCPUS = 0          # set from --n-cpus in main(); 0 -> all cores
_INF = None        # cached DefaultInference


def get_inf():
    """Cached multi-core inference object (big speedup on dispersion fitting)."""
    global _INF
    if _INF is None:
        try:
            from pydeseq2.default_inference import DefaultInference
            n = NCPUS if NCPUS and NCPUS > 0 else (os.cpu_count() or 1)
            _INF = DefaultInference(n_cpus=int(n))
        except Exception as e:
            eprint("   [inference] falling back to default (no n_cpus):", str(e)[:100])
            _INF = False
    return _INF


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
def load_cohort(pb_dir):
    """Load pre-built pseudobulk: counts (groups x genes) + groups metadata."""
    counts = np.load(os.path.join(pb_dir, "counts.npz"), allow_pickle=True)["counts"]
    groups = pd.read_csv(os.path.join(pb_dir, "groups.csv"))
    feats = pd.read_csv(os.path.join(pb_dir, "features.csv"))
    # gene label: prefer symbol, fall back to id
    for c in ["gene_symbol_original", "feature_id_original"]:
        if c in feats.columns:
            genes = np.asarray(feats[c].astype(str), dtype=object)
            break
    else:
        genes = np.array([f"g{i}" for i in range(counts.shape[1])])
    assert counts.shape[0] == len(groups), (counts.shape, len(groups))
    assert counts.shape[1] == len(genes), (counts.shape, len(genes))
    return counts, groups, genes


def case_level(groups):
    labels = set(groups["diagnosis"].unique())
    for cand in ["ASD", "Dup15q", "dup15q", "Case"]:
        if cand in labels:
            return cand
    non = [x for x in labels if str(x).lower() not in ("control", "ctrl", "neurotypical")]
    if len(non) != 1:
        raise SystemExit(f"Cannot identify case label among {labels}")
    return non[0]


def donor_pseudobulk(counts, groups, cell_type, case, min_nuclei):
    """Sum region/sample rows -> one pseudobulk vector per donor for this cell type."""
    sub = groups[groups["cell_type"] == cell_type]
    if len(sub) == 0:
        return None
    rows_by_donor = {}
    for didx, g in sub.groupby("donor_id"):
        nnuc = g["n_nuclei"].sum()
        if nnuc < min_nuclei:
            continue
        vec = counts[g["group_index"].values if "group_index" in g.columns else g.index.values].sum(0)
        rows_by_donor[didx] = (vec, nnuc, g.iloc[0])
    if len(rows_by_donor) < 4:
        return None
    donors = list(rows_by_donor.keys())
    mat = np.vstack([rows_by_donor[d][0] for d in donors]).astype(np.int64)
    meta = pd.DataFrame({
        "donor": donors,
        "diagnosis": [("case" if rows_by_donor[d][2]["diagnosis"] == case else "control") for d in donors],
        "sex": [rows_by_donor[d][2].get("sex", np.nan) for d in donors],
        "age": pd.to_numeric([rows_by_donor[d][2].get("age", np.nan) for d in donors], errors="coerce"),
        "pmi": pd.to_numeric([rows_by_donor[d][2].get("pmi_hours", np.nan) for d in donors], errors="coerce"),
        "rin": pd.to_numeric([rows_by_donor[d][2].get("rin", np.nan) for d in donors], errors="coerce"),
        "region": [str(rows_by_donor[d][2].get("region_source", "NA")) for d in donors],
        "batch": [str(rows_by_donor[d][2].get("sequencing_batch", "NA")) for d in donors],
    })
    return mat, meta


def filter_genes(mat, min_count=10, min_frac=0.5):
    import math
    nd = mat.shape[0]
    min_don = max(3, int(math.ceil(min_frac * nd)))
    keep = (mat.sum(0) >= min_count) & ((mat > 0).sum(0) >= min_don)
    return keep


def _prep_meta(meta, adjust, add_region=False):
    md = meta.copy()
    md["diagnosis"] = pd.Categorical(md["diagnosis"], categories=["control", "case"])
    cov = []
    if adjust:
        for c in ["age", "pmi", "rin"]:
            if c in md and md[c].notna().sum() >= len(md) - 1 and md[c].std(skipna=True) > 0:
                z = (md[c] - md[c].mean()) / md[c].std()
                md[c + "_z"] = z.fillna(0.0)
                cov.append(c + "_z")
        if "sex" in md and md["sex"].notna().any() and md["sex"].nunique() == 2:
            md["sex_b"] = (md["sex"].astype(str) == md["sex"].astype(str).mode().iloc[0]).astype(int)
            cov.append("sex_b")
        # region as a categorical covariate (DESeq2 formula only), guarded for df
        if add_region and "region" in md and md["region"].nunique() >= 2 \
                and (len(md) - md["region"].nunique()) >= 4:
            md["region"] = md["region"].astype(str)
            cov.append("region")
    return md, cov


def deseq_ndeg(mat, meta, genes, alpha, adjust=False, labels=None, add_region=False):
    """n DEG (padj<alpha) for case-vs-control. labels overrides diagnosis (for perms)."""
    if not HAVE_DESEQ:
        return np.nan
    md, cov = _prep_meta(meta, adjust, add_region=add_region)
    if labels is not None:
        md = md.copy()
        md["diagnosis"] = pd.Categorical(labels, categories=["control", "case"])
    if md["diagnosis"].value_counts().min() < 2:
        return np.nan
    design = "~" + (" + ".join(cov + ["diagnosis"]) if cov else "diagnosis")
    idx = list(md["donor"].astype(str))
    cdf = pd.DataFrame(mat, columns=[f"{g}_{i}" for i, g in enumerate(genes)], index=idx)
    md2 = md.set_index(pd.Index(idx))
    try:
        inf = get_inf()
        kw = dict(inference=inf) if inf else {}
        dds = DeseqDataSet(counts=cdf, metadata=md2, design=design, quiet=True, **kw)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["diagnosis", "case", "control"], quiet=True, **kw)
        st.summary()
        padj = st.results_df["padj"].values
        return int(np.nansum(padj < alpha))
    except Exception as e:
        eprint("   [deseq] ", str(e)[:120])
        return np.nan


def deseq_stats(mat, meta, genes, labels=None):
    """Return Wald stat vector (for split-half correlation)."""
    if not HAVE_DESEQ:
        return None
    md, _ = _prep_meta(meta, False)
    if labels is not None:
        md["diagnosis"] = pd.Categorical(labels, categories=["control", "case"])
    if md["diagnosis"].value_counts().min() < 2:
        return None
    idx = list(md["donor"].astype(str))
    cdf = pd.DataFrame(mat, columns=[f"{g}_{i}" for i, g in enumerate(genes)], index=idx)
    md2 = md.set_index(pd.Index(idx))
    try:
        inf = get_inf()
        kw = dict(inference=inf) if inf else {}
        dds = DeseqDataSet(counts=cdf, metadata=md2, design="~diagnosis", quiet=True, **kw)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["diagnosis", "case", "control"], quiet=True, **kw)
        st.summary()
        return st.results_df["stat"].values
    except Exception:
        return None


def linear_ndeg(mat, meta, alpha, adjust=False, labels=None):
    """OLS on log2(CPM+1), gene-by-gene; BH; return n DEG (and the t-stats)."""
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    md, cov = _prep_meta(meta, adjust)
    diag = labels if labels is not None else md["diagnosis"].values
    y_case = (np.asarray(diag) == "case").astype(float)
    if y_case.sum() < 2 or (len(y_case) - y_case.sum()) < 2:
        return np.nan, None
    libsize = mat.sum(1, keepdims=True).astype(float)
    libsize[libsize == 0] = 1.0
    logcpm = np.log2(mat / libsize * 1e6 + 1.0)
    X = [np.ones(len(md)), y_case]
    names = ["const", "case"]
    if adjust:
        for c in cov:
            X.append(md[c].values.astype(float)); names.append(c)
    X = np.column_stack(X)
    pvals, tstats = [], []
    for j in range(logcpm.shape[1]):
        yj = logcpm[:, j]
        if np.ptp(yj) == 0:
            pvals.append(1.0); tstats.append(0.0); continue
        try:
            res = sm.OLS(yj, X).fit()
            pvals.append(res.pvalues[1]); tstats.append(res.tvalues[1])
        except Exception:
            pvals.append(1.0); tstats.append(0.0)
    pvals = np.array(pvals)
    rej = multipletests(pvals, alpha=alpha, method="fdr_bh")[0]
    return int(rej.sum()), np.array(tstats)


def balanced_perm(meta, rng):
    """Permute case/control labels, preserving group sizes."""
    lab = meta["diagnosis"].values.copy()
    return rng.permutation(lab)


# ---------------------------------------------------------------------------
def run_cohort(name, pb_dir, args, rng, all_rows, all_perm):
    import time as _t
    counts, groups, genes = load_cohort(pb_dir)
    case = case_level(groups)
    cts = sorted(groups["cell_type"].dropna().unique())
    eprint(f"\n=== cohort {name}: case label = '{case}', {len(cts)} cell types ===")
    sumpath = os.path.join(args.out, "benchmark_multicohort_summary.csv")
    perpath = os.path.join(args.out, "benchmark_multicohort_perperm.csv")
    # donor composition table: unique donors by diagnosis x region x batch (x cell type)
    try:
        gg = groups.copy()
        gg["diag2"] = np.where(gg["diagnosis"] == case, "case", "control")
        comp = (gg.groupby(["diag2", "region_source", "sequencing_batch", "cell_type"])
                  ["donor_id"].nunique().reset_index(name="n_donors"))
        comp.insert(0, "cohort", name)
        cpath = os.path.join(args.out, "donor_composition.csv")
        hdr = not os.path.exists(cpath)
        comp.to_csv(cpath, mode="a", header=hdr, index=False)
    except Exception as e:
        eprint(f"  [composition] {str(e)[:100]}")
    t0 = None
    for ct in cts:
        try:
            pk = donor_pseudobulk(counts, groups, ct, case, args.min_nuclei)
            if pk is None:
                continue
            mat, meta = pk
            ncase = int((meta["diagnosis"] == "case").sum())
            nctrl = int((meta["diagnosis"] == "control").sum())
            if min(ncase, nctrl) < args.min_donors:
                continue
            keep = filter_genes(mat, args.min_gene_count, args.min_gene_frac)
            if args.max_genes and keep.sum() > args.max_genes:
                tot = np.where(keep, mat.sum(0), -1)
                top = np.argsort(tot)[::-1][:args.max_genes]
                keep = np.zeros(mat.shape[1], bool); keep[top] = True
            mat = mat[:, keep]; g = genes[keep]
            eprint(f"  [{name}/{ct}] donors={ncase}case/{nctrl}ctrl  genes={mat.shape[1]}")

            if args.export_pb:
                ctsafe = str(ct).replace("/", "_").replace(" ", "_")
                pd.DataFrame(mat.T, index=[str(x) for x in g],
                             columns=list(meta["donor"].astype(str))).to_csv(
                    os.path.join(args.export_pb, f"{name}__{ctsafe}__counts.csv"))
                meta.assign(cell_type=ct, cohort=name).to_csv(
                    os.path.join(args.export_pb, f"{name}__{ctsafe}__meta.csv"), index=False)

            _s = _t.time()
            ds_real = deseq_ndeg(mat, meta, g, args.alpha, adjust=False)
            dt = _t.time() - _s
            if t0 is None and dt == dt:
                t0 = dt
                fits = len(cts) * (2 + args.nc_reps)
                eprint(f"    ~{dt:.1f}s / DESeq2 fit; rough ETA for {name}: ~{fits*dt/60:.0f} min")
            ds_real_adj = deseq_ndeg(mat, meta, g, args.alpha, adjust=True)
            ds_real_adj_region = (deseq_ndeg(mat, meta, g, args.alpha, adjust=True, add_region=True)
                                  if args.adjust_region else np.nan)
            lin_real, _ = linear_ndeg(mat, meta, args.alpha, adjust=False)
            lin_real_adj, _ = linear_ndeg(mat, meta, args.alpha, adjust=True)

            # identical permutations for BOTH estimators (same label vector each rep)
            ds_perm, lin_perm = [], []
            for r in range(args.nc_reps):
                lab = balanced_perm(meta, rng)
                ds_perm.append(deseq_ndeg(mat, meta, g, args.alpha, adjust=False, labels=lab))
                lp, _ = linear_ndeg(mat, meta, args.alpha, adjust=False, labels=lab)
                lin_perm.append(lp)
                all_perm.append(dict(cohort=name, cell_type=ct, rep=r,
                                     deseq_perm=ds_perm[-1], linear_perm=lin_perm[-1]))

            # split-half replication via fast donor-level linear-model t-stats
            sh = []
            didx = np.arange(mat.shape[0])
            def _ok(h):
                mm = meta.iloc[h]["diagnosis"]
                return (mm == "case").sum() >= 2 and (mm == "control").sum() >= 2
            for r in range(args.split_reps):
                perm = rng.permutation(didx)
                h1, h2 = perm[: len(perm)//2], perm[len(perm)//2:]
                if not (_ok(h1) and _ok(h2)):
                    continue
                _, s1 = linear_ndeg(mat[h1], meta.iloc[h1].reset_index(drop=True), args.alpha)
                _, s2 = linear_ndeg(mat[h2], meta.iloc[h2].reset_index(drop=True), args.alpha)
                if s1 is not None and s2 is not None:
                    ok = np.isfinite(s1) & np.isfinite(s2)
                    if ok.sum() > 10:
                        sh.append(stats.spearmanr(s1[ok], s2[ok]).correlation)

            from scipy.stats import binomtest
            def summ(a):
                a = np.array([x for x in a if x == x], float)
                if len(a) == 0:
                    return (np.nan, np.nan, np.nan, np.nan, 0, 0)
                k = int(np.sum(a >= 1)); n = len(a)
                return (float(np.mean(a)), float(np.std(a)), float(np.max(a)),
                        k / n, k, n)   # 4th = rejection frequency; k of n splits with >=1 call
            def bci(k, n):
                if n == 0:
                    return (np.nan, np.nan)
                c = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
                return (round(c.low, 3), round(c.high, 3))
            shv = np.array([x for x in sh if x == x], float)
            if shv.size:
                sh_mean = float(np.mean(shv)); sh_sd = float(np.std(shv))
                sem = sh_sd / np.sqrt(shv.size)
                sh_lo, sh_hi = round(sh_mean - 1.96 * sem, 3), round(sh_mean + 1.96 * sem, 3)
            else:
                sh_mean = sh_sd = sh_lo = sh_hi = np.nan

            for est, real, real_adj, real_adj_reg, perm in [
                ("DESeq2", ds_real, ds_real_adj, ds_real_adj_region, ds_perm),
                ("Linear(logCPM)", lin_real, lin_real_adj, np.nan, lin_perm)]:
                m, sd, mx, rf, k, n = summ(perm)
                rlo, rhi = bci(k, n)
                all_rows.append(dict(cohort=name, cell_type=ct, n_case=ncase, n_ctrl=nctrl,
                                 estimator=est, real_deg=real, real_deg_adj=real_adj,
                                 real_deg_adj_region=real_adj_reg,
                                 perm_mean=round(m, 2), perm_sd=round(sd, 2), perm_max=mx,
                                 reject_freq=round(rf, 3), reject_lo=rlo, reject_hi=rhi,
                                 splithalf_r_mean=round(sh_mean, 3) if shv.size else np.nan,
                                 splithalf_r_sd=round(sh_sd, 3) if shv.size else np.nan,
                                 splithalf_lo=sh_lo, splithalf_hi=sh_hi))
            # crash-safe incremental write after every cell type
            pd.DataFrame(all_rows).to_csv(sumpath, index=False)
            pd.DataFrame(all_perm).to_csv(perpath, index=False)
        except Exception as e:
            eprint(f"  !! [{name}/{ct}] failed: {str(e)[:160]} -- skipping")
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\asd-perturb-bridge")
    ap.add_argument("--pb-subdir", default=r"data\interim\E_region_counts_v1")
    ap.add_argument("--cohorts", default="velmeshev,dup15q")
    ap.add_argument("--min-nuclei", type=int, default=20)
    ap.add_argument("--min-donors", type=int, default=4)
    ap.add_argument("--min-gene-count", type=int, default=10)
    ap.add_argument("--min-gene-frac", type=float, default=0.5,
                    help="keep genes expressed (>0) in at least this fraction of donors")
    ap.add_argument("--max-genes", type=int, default=5000,
                    help="cap genes tested per cell type (top by total count); 0 = no cap")
    ap.add_argument("--nc-reps", type=int, default=20)
    ap.add_argument("--split-reps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cpus", type=int, default=0, help="cores for DESeq2 (0 = all)")
    ap.add_argument("--adjust-region", dest="adjust_region", action="store_true", default=True,
                    help="also fit a DESeq2 model adding brain region as a covariate")
    ap.add_argument("--no-adjust-region", dest="adjust_region", action="store_false")
    ap.add_argument("--export-pb", default="", help="dir to export per-cell-type pseudobulk CSVs for the R comparators (edgeR/limma-voom)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    global NCPUS
    NCPUS = args.n_cpus
    if args.export_pb:
        os.makedirs(args.export_pb, exist_ok=True)
    if not HAVE_DESEQ:
        eprint("WARNING: PyDESeq2 not importable -> DESeq2 columns will be NaN:", _DESEQ_ERR)
    eprint(f"[config] cores={NCPUS or (os.cpu_count() or 1)}, nc-reps={args.nc_reps}, "
           f"split-reps={args.split_reps}, min-gene-frac={args.min_gene_frac}, "
           f"max-genes={args.max_genes}")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_rows, all_perm = [], []
    for name in args.cohorts.split(","):
        name = name.strip()
        pb_dir = os.path.join(args.root, args.pb_subdir, name)
        if not os.path.isdir(pb_dir):
            eprint(f"!! missing cohort dir: {pb_dir}"); continue
        run_cohort(name, pb_dir, args, rng, all_rows, all_perm)

    summ = pd.DataFrame(all_rows)
    if summ.empty:
        eprint("No results produced. Check --root / --pb-subdir / --cohorts and that "
               "counts.npz + groups.csv exist for each cohort.")
        sys.exit(2)
    summ.to_csv(os.path.join(args.out, "benchmark_multicohort_summary.csv"), index=False)
    pd.DataFrame(all_perm).to_csv(os.path.join(args.out, "benchmark_multicohort_perperm.csv"), index=False)

    print("\n" + "="*78)
    print("RESULTS TO PASTE BACK (donor-level, two cohorts, identical permutations)")
    print("="*78)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        cols = ["cohort","cell_type","n_case","n_ctrl","estimator","real_deg",
                "real_deg_adj","perm_mean","perm_sd","perm_max","reject_freq",
                "splithalf_r_mean"]
        print(summ[cols].to_string(index=False))

    print("\n----- CROSS-COHORT SUMMARY (median across cell types) -----")
    for name in summ["cohort"].unique():
        for est in summ["estimator"].unique():
            s = summ[(summ.cohort==name) & (summ.estimator==est)]
            if len(s):
                print(f"  {name:10s} {est:14s}  median real DEG={s.real_deg.median():.0f} | "
                      f"median perm DEG={s.perm_mean.median():.2f} | "
                      f"median reject-freq={s.reject_freq.median():.3f} | "
                      f"median split-half r={s.splithalf_r_mean.median():.3f}")
    print("\nWrote:", os.path.join(args.out, "benchmark_multicohort_summary.csv"))
    print("Wrote:", os.path.join(args.out, "benchmark_multicohort_perperm.csv"))
    print(f"[reps: nc-reps={args.nc_reps}, split-reps={args.split_reps}, seed={args.seed}]")


if __name__ == "__main__":
    main()
