#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
power_sim.py
============
Known-truth power / FDR simulation for donor-level pseudobulk differential
expression, to accompany the two-cohort benchmark (TCBB ASD review).

It simulates donor-level pseudobulk counts from a negative-binomial model with a
*known* set of truly differential genes, then runs the same two estimators used
in the benchmark (raw-count DESeq2 and a log-CPM OLS linear model) and measures,
across a grid of donor numbers and effect sizes:

  * sensitivity (power)      = TP / (truly-DE genes)
  * realized FDR             = FP / (genes called)   at BH FDR 0.05
  * mean |log2FC| error on the truly-DE genes (DESeq2)

This answers the reviewer point that detecting a strong dup15q signal does not,
by itself, establish power for the smaller effects expected in idiopathic ASD:
the curves show the donor numbers and effect sizes at which donor-level
inference is and is not adequately powered, with error control held at 0.05.

No R required. Deterministic given --seed.

Run:
  python power_sim.py --sims 20
  python power_sim.py --sims 20 --donors 5,10,15,20 --lfc 0.5,1.0,1.5,2.0
Output: results/power_simulation.csv  (one row per donors x log2FC cell)
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    HAVE_DESEQ = True
except Exception as e:
    HAVE_DESEQ = False
    _ERR = str(e)


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def simulate(n_per_group, n_genes, n_de, lfc, disp, rng):
    """Return counts (2*n_per_group donors x n_genes) and the truly-DE gene mask."""
    n = 2 * n_per_group
    base = rng.gamma(shape=2.0, scale=50.0, size=n_genes) + 5.0   # baseline mean per gene
    de = np.zeros(n_genes, bool)
    de_idx = rng.choice(n_genes, n_de, replace=False); de[de_idx] = True
    # direction of change (half up, half down)
    sign = rng.choice([-1.0, 1.0], n_de)
    fold = np.ones(n_genes); fold[de_idx] = 2.0 ** (sign * lfc)
    size_factor = np.exp(rng.normal(0, 0.25, n))             # donor library-size variation
    is_case = np.array([True] * n_per_group + [False] * n_per_group)
    counts = np.zeros((n, n_genes), dtype=np.int64)
    for i in range(n):
        mu = base * size_factor[i]
        if is_case[i]:
            mu = mu * fold
        # NB: var = mu + disp*mu^2  ->  r = 1/disp, p = r/(r+mu)
        r = 1.0 / disp
        p = r / (r + mu)
        counts[i] = rng.negative_binomial(r, p)
    return counts, is_case, de, de_idx, sign


def deseq_call(counts, is_case, alpha, inf):
    md = pd.DataFrame({"diagnosis": pd.Categorical(
        ["case" if c else "control" for c in is_case], categories=["control", "case"])},
        index=[f"d{i}" for i in range(counts.shape[0])])
    cdf = pd.DataFrame(counts, columns=[f"g{j}" for j in range(counts.shape[1])],
                       index=md.index)
    kw = dict(inference=inf) if inf else {}
    dds = DeseqDataSet(counts=cdf, metadata=md, design="~diagnosis", quiet=True, **kw)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["diagnosis", "case", "control"], quiet=True, **kw)
    st.summary()
    res = st.results_df
    padj = res["padj"].values
    lfc_hat = res["log2FoldChange"].values
    called = np.nan_to_num(padj, nan=1.0) < alpha
    return called, lfc_hat


def linear_call(counts, is_case, alpha):
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    lib = counts.sum(1, keepdims=True).astype(float); lib[lib == 0] = 1
    logcpm = np.log2(counts / lib * 1e6 + 1.0)
    y = is_case.astype(float)
    X = np.column_stack([np.ones(len(y)), y])
    pv = np.ones(counts.shape[1])
    for j in range(counts.shape[1]):
        col = logcpm[:, j]
        if np.ptp(col) == 0:
            continue
        try:
            pv[j] = sm.OLS(col, X).fit().pvalues[1]
        except Exception:
            pv[j] = 1.0
    called = multipletests(pv, alpha=alpha, method="fdr_bh")[0]
    return called


def metrics(called, de):
    called = np.asarray(called, bool)
    tp = int(np.sum(called & de)); fp = int(np.sum(called & ~de))
    n_called = int(np.sum(called)); n_de = int(np.sum(de))
    sens = tp / n_de if n_de else np.nan
    fdr = fp / n_called if n_called else 0.0
    return sens, fdr, n_called


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", default="5,10,15,20", help="donors PER GROUP grid")
    ap.add_argument("--lfc", default="0.5,1.0,1.5,2.0", help="true log2 fold-change grid")
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--n-de", type=int, default=100)
    ap.add_argument("--disp", type=float, default=0.3, help="NB dispersion (var=mu+disp*mu^2)")
    ap.add_argument("--sims", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-cpus", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if not HAVE_DESEQ:
        eprint("ERROR: PyDESeq2 not importable:", _ERR); sys.exit(2)
    os.makedirs(a.out, exist_ok=True)
    inf = DefaultInference(n_cpus=(a.n_cpus if a.n_cpus > 0 else (os.cpu_count() or 1)))
    rng = np.random.default_rng(a.seed)
    donors = [int(x) for x in a.donors.split(",")]
    lfcs = [float(x) for x in a.lfc.split(",")]
    eprint(f"[power] grid donors/group={donors} x lfc={lfcs}, {a.sims} sims each, "
           f"{a.genes} genes ({a.n_de} true DE), disp={a.disp}")

    rows = []
    for nd in donors:
        for lf in lfcs:
            ds_sens, ds_fdr, lin_sens, lin_fdr, lfc_err = [], [], [], [], []
            for s in range(a.sims):
                counts, is_case, de, de_idx, sign = simulate(nd, a.genes, a.n_de, lf, a.disp, rng)
                try:
                    called, lfc_hat = deseq_call(counts, is_case, a.alpha, inf)
                    se, fd, _ = metrics(called, de)
                    ds_sens.append(se); ds_fdr.append(fd)
                    err = np.abs(lfc_hat[de_idx] - sign * lf)
                    lfc_err.append(float(np.nanmean(err)))
                except Exception as e:
                    eprint(f"  [deseq nd={nd} lfc={lf} s={s}] {str(e)[:80]}")
                try:
                    lc = linear_call(counts, is_case, a.alpha)
                    se2, fd2, _ = metrics(lc, de)
                    lin_sens.append(se2); lin_fdr.append(fd2)
                except Exception as e:
                    eprint(f"  [linear nd={nd} lfc={lf} s={s}] {str(e)[:80]}")
            def mean(x): return round(float(np.nanmean(x)), 3) if len(x) else np.nan
            rows.append(dict(donors_per_group=nd, true_log2fc=lf,
                             deseq_power=mean(ds_sens), deseq_fdr=mean(ds_fdr),
                             linear_power=mean(lin_sens), linear_fdr=mean(lin_fdr),
                             deseq_lfc_abs_err=mean(lfc_err), n_sims=a.sims))
            eprint(f"  donors/grp={nd:2d} lfc={lf:.1f} | DESeq2 power={mean(ds_sens)} "
                   f"FDR={mean(ds_fdr)} | linear power={mean(lin_sens)} FDR={mean(lin_fdr)}")
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "power_simulation.csv"), index=False)

    print("\n" + "=" * 72)
    print("POWER SIMULATION (known truth; BH FDR " + str(a.alpha) + ")")
    print("=" * 72)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nWrote:", os.path.join(a.out, "power_simulation.csv"))


if __name__ == "__main__":
    main()
