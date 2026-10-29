#!/usr/bin/env Rscript
# nebula_celllevel.R  --  donor-random-effect mixed model (NEBULA) on CELL-LEVEL
# raw counts, the comparator that pseudobulk cannot provide. Defaults to the
# Velmeshev rawMatrix (genes x cells). NEBULA models within-donor correlation
# directly, so it is the principled middle ground between per-cell and pseudobulk.
#
# IMPORTANT: untested in the author's environment (no R there); it follows the
# documented NEBULA API. Run ONE cell type first as a smoke test, then all:
#   Rscript nebula_celllevel.R --celltype "L2/3"
#   Rscript nebula_celllevel.R            # all cell types
#
# Heavy: ~10^5 cells x ~3x10^4 genes. Expect minutes per cell type.
suppressMessages({library(Matrix); library(nebula)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, d) { i <- which(args == flag); if (length(i)) args[i + 1] else d }
DIR    <- getarg("--dir", "data/raw/velmeshev2019/rawMatrix")  # has matrix.mtx, genes.tsv, barcodes.tsv
METraw <- getarg("--meta", "data/raw/velmeshev2019/meta.tsv")
OUT    <- getarg("--out", "results")
ONE    <- getarg("--celltype", "")     # "" = all
ALPHA  <- as.numeric(getarg("--alpha", "0.05"))
MINCELL<- as.integer(getarg("--min-cells", "25"))   # per donor, as in the benchmark
MINDON <- as.integer(getarg("--min-donors", "4"))
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

cat("Reading matrix (this is the slow step)...\n")
m <- readMM(file.path(DIR, "matrix.mtx"))            # genes x cells
genes <- read.table(file.path(DIR, "genes.tsv"), sep = "\t", header = FALSE, stringsAsFactors = FALSE)
bc    <- read.table(file.path(DIR, "barcodes.tsv"), sep = "\t", header = FALSE, stringsAsFactors = FALSE)[, 1]
rownames(m) <- make.unique(as.character(genes[[ncol(genes)]])); colnames(m) <- bc
meta <- read.table(METraw, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")

# detect columns flexibly
pick <- function(cands) { h <- intersect(tolower(cands), tolower(names(meta)))
  if (!length(h)) stop("missing column among: ", paste(cands, collapse = ", "))
  names(meta)[match(h[1], tolower(names(meta)))] }
cell_col <- names(meta)[1]                         # first column = cell barcode
don_col  <- pick(c("individual", "donor", "donor_id", "sample"))
ct_col   <- pick(c("cluster", "cell_type", "celltype", "clusters"))
dx_col   <- pick(c("diagnosis", "Diagnosis"))
rownames(meta) <- as.character(meta[[cell_col]])
common <- intersect(colnames(m), rownames(meta)); m <- m[, common]; meta <- meta[common, ]
cat(sprintf("Aligned %d cells; donor=%s celltype=%s dx=%s\n", length(common), don_col, ct_col, dx_col))

cts <- if (nzchar(ONE)) ONE else sort(unique(meta[[ct_col]]))
res <- data.frame()
for (ct in cts) {
  sel <- which(meta[[ct_col]] == ct)
  if (!length(sel)) { cat("  (no cells)", ct, "\n"); next }
  sub <- meta[sel, ]; cm <- m[, sel]
  # donors passing min-cells, both groups present
  tab <- table(sub[[don_col]]); keepd <- names(tab)[tab >= MINCELL]
  sub2 <- sub[sub[[don_col]] %in% keepd, ]
  if (length(unique(sub2[[dx_col]])) < 2) { cat("  skip", ct, "(one group)\n"); next }
  gcase <- sum(!duplicated(sub2[[don_col]]) & grepl("ASD|case|Dup", sub2[[dx_col]], ignore.case = TRUE))
  gctrl <- length(unique(sub2[[don_col]])) - gcase
  if (min(gcase, gctrl) < MINDON) { cat("  skip", ct, "(too few donors)\n"); next }
  idx <- which(meta[[ct_col]] == ct & meta[[don_col]] %in% keepd)
  cm <- m[, idx]; sub <- meta[idx, ]
  dx <- factor(ifelse(grepl("ASD|case|Dup", sub[[dx_col]], ignore.case = TRUE), "case", "control"),
               levels = c("control", "case"))
  sid <- as.character(sub[[don_col]])
  pred <- model.matrix(~dx)
  cat(sprintf("[NEBULA] %s: %d cells, %d donors (%d case/%d ctrl)\n",
              ct, ncol(cm), length(unique(sid)), gcase, gctrl))
  out <- tryCatch({
    dg <- group_cell(count = cm, id = sid, pred = pred)        # reorder by subject
    if (is.null(dg)) { re <- nebula(cm, sid, pred = pred) }     # already grouped
    else             { re <- nebula(dg$count, dg$id, pred = dg$pred) }
    pcol <- grep("^p_.*dx", colnames(re$summary), value = TRUE)
    if (!length(pcol)) pcol <- grep("dxcase", colnames(re$summary), value = TRUE)[1]
    padj <- p.adjust(re$summary[[pcol[1]]], method = "BH")
    sum(padj < ALPHA, na.rm = TRUE)
  }, error = function(e) { cat("   NEBULA error:", conditionMessage(e), "\n"); NA })
  res <- rbind(res, data.frame(cohort = basename(dirname(DIR)), cell_type = ct,
                               n_cells = ncol(cm), n_case = gcase, n_ctrl = gctrl,
                               nebula_DEG = out))
  write.csv(res, file.path(OUT, "nebula_summary.csv"), row.names = FALSE)
}
cat("\n==== NEBULA (cell-level mixed model) ====\n"); print(res)
cat("\nWrote:", file.path(OUT, "nebula_summary.csv"), "\n")
