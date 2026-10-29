# asd-single-cell-rigor

Reproducibility materials for the TCBB review *"Computational Single-Cell Genomics of the
Autism Cortex: Methods, Convergence, and the Rigor That Decides Them."*
These scripts regenerate the empirical benchmark (Table V) and the donor-overlap audit on
**open** data; no controlled-access data are used.

## Data

Velmeshev et al. (2019) snRNA-seq cortex, obtained from the UCSC Cell Browser "autism" dataset
(<https://cells.ucsc.edu/?ds=autism>):

- `rawMatrix/matrix.mtx`, `genes.tsv`, `barcodes.tsv`  — raw UMI counts (genes x cells)
- `meta.tsv`                                           — per-cell metadata (cluster, individual, diagnosis, ...)
- a log-normalised `.h5ad` of the same cells           — used only for the per-cell Wilcoxon test

Donor metadata (`donor_meta.csv`: donor, cohort, diagnosis, sex, age, pmi, rin) are derived from
`meta.tsv`; the dup15q (Perez 2025) donor table is likewise from its UCSC deposit.

## Environment

```
python -m venv .venv && . .venv/bin/activate      # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```
Key versions used for the reported numbers: PyDESeq2 0.5.4, scanpy >= 1.10, statsmodels, scipy, numpy, pandas.

## Reproducing the benchmark (Table V)

Five estimators across both cohorts, on identical donor sets and identical label permutations.

```
# 1) Python: DESeq2 + log-CPM linear, 100-permutation calibration with 95% CIs,
#    region-adjusted model, donor-composition table, and pseudobulk export for R:
python benchmark_multicohort.py --nc-reps 100 --max-genes 4000 --export-pb pbexport

# 2) Python: known-truth power/FDR vs donor number and effect size:
python power_sim.py --sims 50

# 3) Python: shared-control sensitivity (cross-cohort confound, 0..7 shared controls):
python sharedcontrol_sensitivity.py --sims 500

# 4) R: genuine edgeR quasi-likelihood + limma-voom on the exported pseudobulk:
Rscript mixed_models.R --export pbexport --perm 50 --split 10

# 5) R: NEBULA cell-level mixed model (donor random effect), Velmeshev raw matrix:
Rscript nebula_celllevel.R
```

Outputs land in `results/`:
- `benchmark_multicohort_summary.csv` — per-cell-type DESeq2 / linear DEG, rejection freq + 95% CIs, split-half
- `benchmark_multicohort_perperm.csv` — per-permutation discovery counts
- `donor_composition.csv` — donors by diagnosis x region x batch x cell type
- `power_simulation.csv`, `sharedcontrol_sensitivity.csv`
- `mixedmodels_summary.csv` — edgeR-QL + limma-voom; `nebula_summary.csv` — NEBULA

## Environments

- Python: PyDESeq2 0.5.4, scanpy >= 1.10, statsmodels, scipy, numpy, pandas (see `requirements.txt`).
- R (>= 4.5): `edgeR`, `limma`, `nebula`, `Matrix`.

## Method summary

- Pseudobulk = sum of raw UMIs per donor within each cell type; donor is the unit of replication.
- Estimators: DESeq2 and edgeR-QL (negative-binomial count), limma-voom (precision-weighted empirical Bayes),
  log-CPM OLS linear; plus cell-level NEBULA (donor random effect) and a per-cell Wilcoxon reference.
- Benjamini–Hochberg FDR within each cell type; alpha = 0.05; genes expressed in >= 50% of donors, top 5000 by count.
- Randomized-label control (100 perms, identical across estimators) and split-half (10 partitions); fixed seed (0).

All results are deterministic given the seed. Issues / questions: open an issue on this repository.
