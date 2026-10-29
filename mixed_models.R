#!/usr/bin/env Rscript
# mixed_models.R  --  genuine edgeR quasi-likelihood and limma-voom comparators
# on the donor-level pseudobulk exported by benchmark_multicohort.py --export-pb.
#
# Adds the two "standard comparators" reviewers asked for, run on the SAME donor
# pseudobulk as the Python DESeq2/linear benchmark, with a matched randomized-
# label calibration control and split-half replication. (NEBULA, the cell-level
# mixed model, is in the separate nebula_celllevel.R.)
#
# Usage (from the folder that contains the export dir):
#   Rscript mixed_models.R --export pbexport --out results --perm 100 --split 10 --seed 0
#
# Writes: results/mixedmodels_summary.csv  and  results/mixedmodels_perperm.csv
suppressMessages({library(edgeR); library(limma)})

## ---- args ----
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) { i <- which(args == flag); if (length(i)) args[i + 1] else default }
EXPORT <- getarg("--export", "pbexport")
OUT    <- getarg("--out", "results")
NPERM  <- as.integer(getarg("--perm", "100"))
NSPLIT <- as.integer(getarg("--split", "10"))
ALPHA  <- as.numeric(getarg("--alpha", "0.05"))
SEED   <- as.integer(getarg("--seed", "0"))
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
set.seed(SEED)

counts_files <- list.files(EXPORT, pattern = "__counts\\.csv$", full.names = TRUE)
if (!length(counts_files)) stop("No *__counts.csv in ", EXPORT, " (run benchmark_multicohort.py --export-pb first)")

## ---- per-estimator DEG count for a given label vector ----
fit_counts <- function(cnt, grp, covdf = NULL, alpha = ALPHA, want_t = FALSE) {
  grp <- factor(grp, levels = c("control", "case"))
  if (min(table(grp)) < 2) return(list(edger = NA, voom = NA, t = NULL))
  if (is.null(covdf)) design <- model.matrix(~grp)
  else                design <- model.matrix(as.formula(paste0("~", paste(colnames(covdf), collapse = "+"), "+grp")),
                                             data = cbind(covdf, grp = grp))
  ci <- which(colnames(design) == "grpcase"); if (!length(ci)) return(list(edger = NA, voom = NA, t = NULL))
  dge <- DGEList(counts = cnt)
  keep <- filterByExpr(dge, design); dge <- dge[keep, , keep.lib.sizes = FALSE]
  if (nrow(dge) < 10) return(list(edger = NA, voom = NA, t = NULL))
  dge <- calcNormFactors(dge)
  # edgeR quasi-likelihood
  edg <- tryCatch({
    d <- estimateDisp(dge, design); f <- glmQLFit(d, design)
    tt <- topTags(glmQLFTest(f, coef = ci), n = Inf)$table
    sum(tt$FDR < alpha, na.rm = TRUE)
  }, error = function(e) NA)
  # limma-voom (genuine: precision weights + empirical Bayes)
  vv <- tryCatch({
    v <- voom(dge, design); vf <- eBayes(lmFit(v, design))
    tt2 <- topTable(vf, coef = ci, number = Inf, sort.by = "none")
    list(n = sum(tt2$adj.P.Val < alpha, na.rm = TRUE),
         t = if (want_t) setNames(tt2$t, rownames(tt2)) else NULL)
  }, error = function(e) list(n = NA, t = NULL))
  list(edger = edg, voom = vv$n, t = vv$t)
}

summ <- data.frame(); pp <- data.frame()
for (cf in counts_files) {
  base <- sub("__counts\\.csv$", "", basename(cf))
  mf <- file.path(EXPORT, paste0(base, "__meta.csv"))
  if (!file.exists(mf)) next
  raw <- read.csv(cf, check.names = FALSE)                 # 1st col = gene names (may duplicate)
  rn  <- make.unique(as.character(raw[[1]]))
  cnt <- as.matrix(raw[, -1, drop = FALSE]); rownames(cnt) <- rn
  meta <- read.csv(mf, check.names = FALSE)
  meta <- meta[match(colnames(cnt), as.character(meta$donor)), ]
  grp <- meta$diagnosis
  nc <- sum(grp == "case"); nk <- sum(grp == "control")
  if (min(nc, nk) < 3) next
  coh <- meta$cohort[1]; ct <- meta$cell_type[1]
  cat(sprintf("[%s/%s] %d case / %d ctrl, %d genes\n", coh, ct, nc, nk, nrow(cnt)))
  # covariates (standardize numerics; sex as factor)
  cov <- data.frame(row.names = colnames(cnt))
  for (v in c("age", "pmi", "rin")) if (v %in% names(meta) && sd(meta[[v]], na.rm = TRUE) > 0) {
    z <- scale(meta[[v]])[, 1]; z[is.na(z)] <- 0; cov[[v]] <- z
  }
  if ("sex" %in% names(meta) && length(unique(meta$sex)) == 2) cov$sex <- factor(meta$sex)
  if (!ncol(cov)) cov <- NULL

  real <- fit_counts(cnt, grp, NULL, want_t = TRUE)
  real_adj <- fit_counts(cnt, grp, cov)
  # randomized-label calibration (identical permuted labels for both estimators)
  ed_k <- 0; vo_k <- 0; ed_n <- 0; vo_n <- 0
  for (r in seq_len(NPERM)) {
    pg <- sample(grp)
    fc <- fit_counts(cnt, pg, NULL)
    if (!is.na(fc$edger)) { ed_n <- ed_n + 1; if (fc$edger >= 1) ed_k <- ed_k + 1 }
    if (!is.na(fc$voom))  { vo_n <- vo_n + 1; if (fc$voom  >= 1) vo_k <- vo_k + 1 }
    pp <- rbind(pp, data.frame(cohort = coh, cell_type = ct, rep = r,
                               edger_perm = fc$edger, voom_perm = fc$voom))
  }
  # split-half (voom moderated t correlation)
  shv <- c()
  for (r in seq_len(NSPLIT)) {
    idx <- sample(ncol(cnt)); h1 <- idx[1:(length(idx) %/% 2)]; h2 <- idx[(length(idx) %/% 2 + 1):length(idx)]
    ok <- function(h) min(sum(grp[h] == "case"), sum(grp[h] == "control")) >= 2
    if (!(ok(h1) && ok(h2))) next
    t1 <- fit_counts(cnt[, h1], grp[h1], NULL, want_t = TRUE)$t
    t2 <- fit_counts(cnt[, h2], grp[h2], NULL, want_t = TRUE)$t
    if (!is.null(t1) && !is.null(t2)) {
      g <- intersect(names(t1), names(t2)); if (length(g) > 10)
        shv <- c(shv, suppressWarnings(cor(t1[g], t2[g], method = "spearman")))
    }
  }
  bci <- function(k, n) if (n == 0) c(NA, NA) else { ci <- binom.test(k, n)$conf.int; round(c(ci[1], ci[2]), 3) }
  ed_ci <- bci(ed_k, ed_n); vo_ci <- bci(vo_k, vo_n)
  summ <- rbind(summ,
    data.frame(cohort = coh, cell_type = ct, n_case = nc, n_ctrl = nk,
               estimator = "edgeR-QL", real_deg = real$edger, real_deg_adj = real_adj$edger,
               reject_freq = ifelse(ed_n, round(ed_k / ed_n, 3), NA), reject_lo = ed_ci[1], reject_hi = ed_ci[2],
               splithalf_r = ifelse(length(shv), round(mean(shv), 3), NA)),
    data.frame(cohort = coh, cell_type = ct, n_case = nc, n_ctrl = nk,
               estimator = "limma-voom", real_deg = real$voom, real_deg_adj = real_adj$voom,
               reject_freq = ifelse(vo_n, round(vo_k / vo_n, 3), NA), reject_lo = vo_ci[1], reject_hi = vo_ci[2],
               splithalf_r = ifelse(length(shv), round(mean(shv), 3), NA)))
  write.csv(summ, file.path(OUT, "mixedmodels_summary.csv"), row.names = FALSE)
  write.csv(pp,   file.path(OUT, "mixedmodels_perperm.csv"), row.names = FALSE)
}
cat("\n==== edgeR-QL / limma-voom summary (medians across cell types) ====\n")
for (coh in unique(summ$cohort)) for (es in c("edgeR-QL", "limma-voom")) {
  s <- summ[summ$cohort == coh & summ$estimator == es, ]
  if (nrow(s)) cat(sprintf("  %-10s %-11s median real DEG=%.0f | median reject-freq=%.3f | median split-half r=%.3f\n",
                           coh, es, median(s$real_deg, na.rm = TRUE), median(s$reject_freq, na.rm = TRUE),
                           median(s$splithalf_r, na.rm = TRUE)))
}
cat("\nWrote:", file.path(OUT, "mixedmodels_summary.csv"), "and mixedmodels_perperm.csv\n")