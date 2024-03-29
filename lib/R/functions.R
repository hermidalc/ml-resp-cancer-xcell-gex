suppressPackageStartupMessages(library("Biobase"))

r_lib_dir <- dirname(sys.frame(1)$ofile)
source(paste(r_lib_dir, "fcbf.R", sep="/"))

eset_class_labels <- function(eset, samples=NULL) {
    if (!is.null(samples)) {
        return(eset$Class[c(samples)])
    } else {
        return(eset$Class)
    }
}

eset_feature_idxs <- function(eset, features) {
    return(as.integer(which(rownames(eset) %in% features)) - 1)
}

eset_feature_annots <- function(eset, annots=annots, features=NULL) {
    if (!is.null(features)) {
        annots <- as.matrix(fData(eset)[c(features), c(annots), drop=FALSE])
    } else {
        annots <- as.matrix(fData(eset)[, c(annots), drop=FALSE])
    }
    annots[is.na(annots)] <- ""
    return(annots)
}

data_nzero_col_idxs <- function(X) {
    return(as.integer(which(colSums(X) > 0)) - 1)
}

data_nzero_sd_col_idxs <- function(X) {
    return(as.integer(which(apply(X, 2, function(c) sd(c) != 0))) - 1)
}

data_nzero_var_col_idxs <- function(X, freqCut=95/5, uniqueCut=1) {
    return(sort(setdiff(
        seq_len(ncol(X)),
        caret::nearZeroVar(X, freqCut=freqCut, uniqueCut=uniqueCut
    ))) - 1)
}

data_corr_col_idxs <- function(X, cutoff=0.5) {
    return(sort(caret::findCorrelation(cor(X), cutoff=cutoff)) - 1)
}

deseq2_vst_fit <- function(
    X, y, y_meta=NULL, blind=FALSE, fit_type="local", model_batch=FALSE
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    geo_means <- exp(rowMeans(log(counts)))
    if (!is.null(y_meta) && model_batch) {
        y_meta$Batch <- as.factor(y_meta$Batch)
        y_meta$Class <- as.factor(y_meta$Class)
        dds <- DESeqDataSetFromMatrix(
            counts, as.data.frame(y_meta), ~Batch + Class
        )
    } else {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(Class=factor(y)), ~Class
        )
    }
    dds <- estimateSizeFactors(dds, quiet=TRUE)
    dds <- estimateDispersions(dds, fitType=fit_type, quiet=TRUE)
    vsd <- varianceStabilizingTransformation(dds, blind=blind, fitType=fit_type)
    return(list(
        t(as.matrix(assay(vsd))), geo_means, sizeFactors(dds),
        dispersionFunction(dds)
    ))
}

deseq2_vst_transform <- function(
    X, geo_means, size_factors, disp_func, fit_type="local"
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    dds <- DESeqDataSetFromMatrix(
        counts, data.frame(row.names=seq(1, ncol(counts))), ~1
    )
    dds <- estimateSizeFactors(dds, geoMeans=geo_means, quiet=TRUE)
    suppressMessages(dispersionFunction(dds) <- disp_func)
    vsd <- varianceStabilizingTransformation(dds, blind=FALSE, fitType=fit_type)
    return(t(as.matrix(assay(vsd))))
}

deseq2_feature_score <- function(
    X, y, y_meta=NULL, lfc=0, blind=FALSE, fit_type="local", model_batch=FALSE,
    n_threads=1
) {
    suppressPackageStartupMessages(library("DESeq2"))
    suppressPackageStartupMessages(library("BiocParallel"))
    if (n_threads > 1) {
        register(MulticoreParam(workers=n_threads))
        parallel <- TRUE
    } else {
        register(SerialParam())
        parallel <- FALSE
    }
    counts <- t(X)
    geo_means <- exp(rowMeans(log(counts)))
    if (!is.null(y_meta) && model_batch) {
        y_meta$Batch <- as.factor(y_meta$Batch)
        y_meta$Class <- as.factor(y_meta$Class)
        dds <- DESeqDataSetFromMatrix(
            counts, as.data.frame(y_meta), ~Batch + Class
        )
    } else {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(Class=factor(y)), ~Class
        )
    }
    dds <- DESeq(dds, fitType=fit_type, parallel=parallel, quiet=TRUE)
    suppressMessages(results <- as.data.frame(lfcShrink(
        dds, coef=length(resultsNames(dds)), type="apeglm", lfcThreshold=lfc,
        svalue=TRUE, parallel=parallel, quiet=TRUE
    )))
    # results <- as.data.frame(results(
    #     dds, name=resultsNames(dds)[length(resultsNames(dds))],
    #     lfcThreshold=lfc, altHypothesis="greaterAbs", pAdjustMethod="BH"
    # ))
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    vsd <- varianceStabilizingTransformation(dds, blind=blind, fitType=fit_type)
    return(list(
        results$svalue, t(as.matrix(assay(vsd))), geo_means, sizeFactors(dds),
        dispersionFunction(dds)
    ))
}

edger_filterbyexpr_mask <- function(X, y, y_meta=NULL, model_batch=FALSE) {
    suppressPackageStartupMessages(library("edgeR"))
    dge <- DGEList(counts=t(X))
    if (!is.null(y_meta) && model_batch) {
        y_meta$Batch <- as.factor(y_meta$Batch)
        y_meta$Class <- as.factor(y_meta$Class)
        design <- model.matrix(~Batch + Class, data=y_meta)
    } else {
        design <- model.matrix(~factor(y))
    }
    return(filterByExpr(dge, design))
}

edger_logcpm_transform <- function(X, prior_count=1) {
    return(t(edgeR::cpm(t(X), log=TRUE, prior.count=prior_count)))
}

# adapted from edgeR::calcNormFactors source code
edger_tmm_ref_column <- function(counts, lib.size=colSums(counts), p=0.75) {
    y <- t(t(counts) / lib.size)
    f <- apply(y, 2, function(x) quantile(x, p=p))
    ref_column <- which.min(abs(f - mean(f)))
}

edger_tmm_logcpm_fit <- function(X, prior_count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(t(log_cpm), ref_sample))
}

edger_tmm_logcpm_transform <- function(X, ref_sample, prior_count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    counts <- cbind(counts, ref_sample)
    colnames(counts) <- NULL
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    log_cpm <- log_cpm[, -ncol(log_cpm)]
    return(t(log_cpm))
}

edger_feature_score <- function(
    X, y, y_meta=NULL, lfc=0, robust=TRUE, prior_count=1, model_batch=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    if (!is.null(y_meta) && model_batch) {
        y_meta$Batch <- as.factor(y_meta$Batch)
        y_meta$Class <- as.factor(y_meta$Class)
        design <- model.matrix(~Batch + Class, data=y_meta)
    } else {
        design <- model.matrix(~factor(y))
    }
    dge <- estimateDisp(dge, design, robust=robust)
    fit <- glmQLFit(dge, design, robust=robust)
    if (lfc == 0) {
        glt <- glmQLFTest(fit, coef=ncol(design))
    } else {
        glt <- glmTreat(fit, coef=ncol(design), lfc=lfc)
    }
    results <- as.data.frame(topTags(
        glt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$PValue, results$FDR, t(log_cpm), ref_sample))
}

limma_voom_feature_score <- function(
    X, y, y_meta=NULL, lfc=0, robust=TRUE, prior_count=1, model_batch=FALSE,
    model_dupcor=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    if (!is.null(y_meta) && (model_batch || model_dupcor)) {
        if (model_batch) {
            formula <- ~Batch + Class
            y_meta$Batch <- as.factor(y_meta$Batch)
        } else {
            formula <- ~Class
        }
        y_meta$Class <- as.factor(y_meta$Class)
        design <- model.matrix(formula, data=y_meta)
        v <- voom(dge, design)
        if (model_dupcor) {
            y_meta$Group <- as.factor(y_meta$Group)
            suppressMessages(
                dupcor <- duplicateCorrelation(v, design, block=y_meta$Group)
            )
            v <- voom(
                dge, design, block=y_meta$Group, correlation=dupcor$consensus
            )
            suppressMessages(
                dupcor <- duplicateCorrelation(v, design, block=y_meta$Group)
            )
            fit <- lmFit(
                v, design, block=y_meta$Group, correlation=dupcor$consensus
            )
        } else {
            fit <- lmFit(v, design)
        }
    } else {
        design <- model.matrix(~factor(y))
        v <- voom(dge, design)
        fit <- lmFit(v, design)
    }
    fit <- treat(fit, lfc=lfc, robust=robust)
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$P.Value, results$adj.P.Val, t(log_cpm), ref_sample))
}

dream_voom_feature_score <- function(
    X, y, y_meta, lfc=0, prior_count=1, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    suppressPackageStartupMessages(library("variancePartition"))
    suppressPackageStartupMessages(library("BiocParallel"))
    if (n_threads > 1) {
        register(MulticoreParam(workers=n_threads))
    } else {
        register(SerialParam())
    }
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    if (model_batch) {
        formula <- ~Batch + Class + (1|Group)
        y_meta$Batch <- as.factor(y_meta$Batch)
    } else {
        formula <- ~Class + (1|Group)
    }
    y_meta$Class <- as.factor(y_meta$Class)
    y_meta$Group <- as.factor(y_meta$Group)
    invisible(capture.output(
        v <- voomWithDreamWeights(dge, formula, y_meta)
    ))
    invisible(capture.output(
        fit <- dream(v, formula, y_meta, suppressWarnings=TRUE)
    ))
    results <- topTable(
        fit, coef=ncol(design), lfc=lfc, number=Inf, adjust.method="BH",
        sort.by="none"
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$P.Value, results$adj.P.Val, t(log_cpm), ref_sample))
}

limma_feature_score <- function(
    X, y, y_meta=NULL, lfc=0, robust=FALSE, trend=FALSE, model_batch=FALSE
) {
    suppressPackageStartupMessages(library("limma"))
    if (!is.null(y_meta) && model_batch) {
        y_meta$Batch <- as.factor(y_meta$Batch)
        y_meta$Class <- as.factor(y_meta$Class)
        design <- model.matrix(~Batch + Class, data=y_meta)
    } else {
        design <- model.matrix(~factor(y))
    }
    fit <- lmFit(t(X), design)
    fit <- treat(fit, lfc=lfc, robust=robust, trend=trend)
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(results$P.Value, results$adj.P.Val))
}

# adapted from limma::removeBatchEffect source code
limma_remove_ba_fit <- function(X, y_meta, preserve_design=TRUE) {
    suppressPackageStartupMessages(library("limma"))
    batch <- y_meta$Batch
    y_meta$Batch <- NULL
    if (preserve_design) {
        y_meta$Class <- as.factor(y_meta$Class)
        design <- model.matrix(~Class, data=y_meta)
    } else {
        design <- matrix(1, ncol(t(X)), 1)
    }
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    fit <- lmFit(t(X), cbind(design, batch))
    beta <- fit$coefficients[, -seq_len(ncol(design)), drop=FALSE]
    beta[is.na(beta)] <- 0
    return(beta)
}

limma_remove_ba_transform <- function(X, y_meta, beta) {
    batch <- y_meta$Batch
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    return(t(t(X) - beta %*% t(batch)))
}

fcbf_feature_idxs <- function(X, y, threshold=0) {
    results <- Biocomb::select.fast.filter(
        cbind(X, as.factor(y)), disc.method="MDL", threshold=threshold
    )
    results <- results[order(results$NumberFeature), , drop=FALSE]
    return(list(results$NumberFeature - 1, results$Information.Gain))
}

cfs_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    feature_idxs <- FSelector::cfs(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y))
    )
    return(as.integer(feature_idxs) - 1)
}

gain_ratio_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::gain.ratio(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

sym_uncert_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::symmetrical.uncertainty(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

relieff_feature_score <- function(X, y, num_neighbors=10, sample_size=5) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)),
        neighbours.count=num_neighbors, sample.size=sample_size
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(results$attr_importance)
}
