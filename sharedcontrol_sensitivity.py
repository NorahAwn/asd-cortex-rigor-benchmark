#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sharedcontrol_sensitivity.py
============================
Quantifies how reusing control donors across two cohorts inflates the apparent
cross-cohort concordance of case-vs-control signatures, EVEN WHEN the two
cohorts' true case effects are independent (no shared biology).

This converts the manuscript's "candidate shared controls" caution into a
quantified sensitivity analysis: for k = 0..K shared control donors it reports
the Pearson correlation between the two cohorts' per-gene signatures, with 95%
confidence intervals over repeated simulations.

Model (per gene, independent across genes):
  expression = gene baseline + donor random effect (+ case effect if a case).
  Case effects are drawn INDEPENDENTLY for the two cohorts, so under k=0 the
  expected signature correlation is 0; any positive correlation at k>0 is the
  shared-control confound alone. Donors listed as "shared" contribute the SAME
  donor random-effect vector to both cohorts' control means.

No external dependencies beyond numpy/pandas. Deterministic given --seed.

Run:
  python sharedcontrol_sensitivity.py
  python sharedcontrol_sensitivity.py --n-ctrl 16 --max-shared 7 --sims 500
Output: results/sharedcontrol_sensitivity.csv
"""
import os, argparse
import numpy as np
import pandas as pd


def one_sim(G, n_caseA, n_caseB, n_ctrlA, n_ctrlB, k_shared,
            sigma_donor, sigma_case, rng):
    # gene baselines cancel in the case-minus-control signature, so omit them.
    def donor_block(n):
        return rng.normal(0, sigma_donor, size=(n, G))
    # shared control donors (same random-effect vectors in both cohorts)
    shared = donor_block(k_shared)
    ctrlA = np.vstack([shared, donor_block(n_ctrlA - k_shared)]) if k_shared else donor_block(n_ctrlA)
    ctrlB = np.vstack([shared, donor_block(n_ctrlB - k_shared)]) if k_shared else donor_block(n_ctrlB)
    # cases: independent donor effects + independent per-cohort case effect
    caseA = donor_block(n_caseA) + rng.normal(0, sigma_case, size=(1, G))
    caseB = donor_block(n_caseB) + rng.normal(0, sigma_case, size=(1, G))
    sigA = caseA.mean(0) - ctrlA.mean(0)
    sigB = caseB.mean(0) - ctrlB.mean(0)
    if np.std(sigA) == 0 or np.std(sigB) == 0:
        return np.nan
    return float(np.corrcoef(sigA, sigB)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--n-case-a", type=int, default=15)   # Velmeshev-like
    ap.add_argument("--n-case-b", type=int, default=11)   # dup15q-like
    ap.add_argument("--n-ctrl", type=int, default=16)     # both ~16 controls
    ap.add_argument("--max-shared", type=int, default=7)  # up to the 7 candidates
    ap.add_argument("--sigma-donor", type=float, default=1.0)
    ap.add_argument("--sigma-case", type=float, default=0.3,
                    help="independent per-cohort case effect SD (0 = pure confound)")
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows = []
    for k in range(0, min(a.max_shared, a.n_ctrl) + 1):
        rs = [one_sim(a.genes, a.n_case_a, a.n_case_b, a.n_ctrl, a.n_ctrl, k,
                      a.sigma_donor, a.sigma_case, rng) for _ in range(a.sims)]
        rs = np.array([x for x in rs if x == x])
        mean = float(np.mean(rs)); sd = float(np.std(rs))
        lo = float(np.percentile(rs, 2.5)); hi = float(np.percentile(rs, 97.5))
        rows.append(dict(k_shared_controls=k, mean_signature_r=round(mean, 3),
                         sd=round(sd, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                         n_sims=len(rs)))
        print(f"  k={k:2d} shared controls | mean signature r = {mean:+.3f} "
              f"[95% {lo:+.3f}, {hi:+.3f}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "sharedcontrol_sensitivity.csv"), index=False)
    print("\n" + "=" * 64)
    print("SHARED-CONTROL SENSITIVITY  (independent true case effects)")
    print("=" * 64)
    print(df.to_string(index=False))
    print(f"\nControls per cohort = {a.n_ctrl}; cases A/B = {a.n_case_a}/{a.n_case_b}; "
          f"sigma_case = {a.sigma_case}")
    print("Wrote:", os.path.join(a.out, "sharedcontrol_sensitivity.csv"))


if __name__ == "__main__":
    main()
