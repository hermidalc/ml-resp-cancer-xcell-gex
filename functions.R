suppressPackageStartupMessages(library("Biobase"))

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
        1:ncol(X), caret::nearZeroVar(X, freqCut=freqCut, uniqueCut=uniqueCut)
    )) - 1)
}

data_corr_col_idxs <- function(X, cutoff=0.5) {
    return(sort(caret::findCorrelation(cor(X), cutoff=cutoff)) - 1)
}

edger_filterbyexpr_mask <- function(X, y) {
    suppressPackageStartupMessages(library("edgeR"))
    return(filterByExpr(DGEList(counts=t(X), group=y)))
}

edger_logcpm_transform <- function(X, prior.count=1) {
    return(t(edgeR::cpm(t(X), log=TRUE, prior.count=prior.count)))
}

# from edgeR codebase
edger_tmm_ref_column <- function(counts, lib.size=colSums(counts), p=0.75) {
    y <- t(t(counts) / lib.size)
    f <- apply(y, 2, function(x) quantile(x, p=p))
    ref_column <- which.min(abs(f - mean(f)))
}

edger_tmm_ref_sample <- function(X) {
    counts <- t(X)
    return(counts[, edger_tmm_ref_column(counts=counts)])
}

edger_tmm_logcpm_transform <- function(X, ref_sample=NULL, prior.count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    if (is.null(ref_sample)) {
        dge <- DGEList(counts=counts)
        dge <- calcNormFactors(dge, method="TMM")
        log_cpm <- cpm(dge, log=TRUE, prior.count=prior.count)
        ref_sample <- counts[, edger_tmm_ref_column(counts=counts)]
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts)
        dge <- calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
        log_cpm <- cpm(dge, log=TRUE, prior.count=prior.count)
        log_cpm <- log_cpm[, -ncol(log_cpm)]
    }
    return(list(t(log_cpm), ref_sample))
}

limma_voom_feature_score <- function(X, y, prior.count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    counts <- t(X)
    dge <- DGEList(counts=counts, group=y)
    dge <- calcNormFactors(dge, method="TMM")
    design <- model.matrix(~0 + factor(y))
    colnames(design) <- c("Class0", "Class1")
    v <- voom(dge, design)
    fit <- lmFit(v, design)
    fit <- contrasts.fit(fit, makeContrasts(
        Class1VsClass0=Class1-Class0, levels=design
    ))
    fit <- eBayes(fit)
    results <- topTableF(fit, number=Inf, adjust.method="BH", sort.by="none")
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior.count)
    return(list(results$F, results$adj.P.Val, t(log_cpm)))
}

limma_feature_score <- function(X, y, robust=FALSE, trend=FALSE) {
    suppressPackageStartupMessages(library("limma"))
    design <- model.matrix(~0 + factor(y))
    colnames(design) <- c("Class0", "Class1")
    fit <- lmFit(t(X), design)
    fit <- contrasts.fit(fit, makeContrasts(
        Class1VsClass0=Class1-Class0, levels=design
    ))
    fit <- eBayes(fit, robust=robust, trend=trend)
    results <- topTableF(fit, number=Inf, adjust.method="BH", sort.by="none")
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(results$F, results$adj.P.Val))
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
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y))
    )
    return(as.integer(feature_idxs) - 1)
}

gain_ratio_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::gain.ratio(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

sym_uncert_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::symmetrical.uncertainty(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

relieff_feature_score <- function(X, y, num.neighbors=10, sample.size=5) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=as.factor(y)),
        neighbours.count=num.neighbors, sample.size=sample.size
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(results$attr_importance)
}
