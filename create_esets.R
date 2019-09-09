#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("WGCNA"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument(
    "--dataset", type="character", nargs="+", help="datasets"
)
parser$add_argument(
    "--data-type", type="character", nargs="+", help="data type"
)
parser$add_argument(
    "--norm-meth", type="character", nargs="+", help="normalization method"
)
parser$add_argument(
    "--feat-type", type="character", nargs="+", help="feature type"
)
parser$add_argument(
    "--gex-collapse-meth", type="character", default="MaxMean",
    help="gex collapse rows method"
)
args <- parser$parse_args()
all_dataset_names <- dataset_names
if (!is.null(args$dataset)) {
    dataset_names <- intersect(dataset_names, args$dataset)
}
if (!is.null(args$data_type)) {
    data_types <- intersect(data_types, args$data_type)
}
if (!is.null(args$norm_meth)) {
    norm_methods <- intersect(norm_methods, args$norm_meth)
}
if (!is.null(args$feat_type)) {
    feat_types <- intersect(feat_types, args$feat_type)
}
for (dataset_name in dataset_names) {
    for (data_type in data_types) {
        suffixes <- c(data_type)
        pdata_file_basename <- paste0(
            c(dataset_name, suffixes, "meta"), collapse="_"
        )
        pdata_file <- paste0("data/", pdata_file_basename, ".txt")
        if (data_type %in% c("tmb", "xcell")) {
            exprs_file_basename <- paste0(
                c(dataset_name, suffixes), collapse="_"
            )
            exprs_file <- paste0("data/", exprs_file_basename, ".txt")
            if (file.exists(pdata_file) && file.exists(exprs_file)) {
                cat(" Loading:", pdata_file_basename, "\n")
                pdata <- read.delim(pdata_file, row.names=1)
                rownames(pdata) <- make.names(rownames(pdata))
                pdata <- cbind(pdata, rep(
                    which(dataset_name == all_dataset_names)[1],
                    nrow(pdata)
                ))
                colnames(pdata)[ncol(pdata)] <- "Batch"
                pdata <- pdata[!is.na(pdata$Class), ]
                eset_name <- paste0(
                    c("eset", exprs_file_basename), collapse="_"
                )
                cat("Creating:", eset_name, "\n")
                exprs <- read.delim(exprs_file, row.names=1)
                if (nrow(pdata) > ncol(exprs)) {
                    pdata <- pdata[colnames(exprs), , drop=FALSE]
                } else if (nrow(pdata) < ncol(exprs)) {
                    exprs <- exprs[, rownames(pdata), drop=FALSE]
                }
                if (data_type == "tmb") {
                    exprs[wes_tmb_feat_name, ] <-
                        log2(exprs[wes_tmb_feat_name, ] + 1)
                }
                eset <- ExpressionSet(
                    assayData=as.matrix(exprs),
                    phenoData=AnnotatedDataFrame(pdata)
                )
                assign(eset_name, eset)
                save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
            }
        } else if (
            data_type %in% c("gex", "gex_cyt", "gex_cyt_tmb", "gex_tmb")
        ) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    exprs_file_basename <- paste0(
                        c(dataset_name, suffixes), collapse="_"
                    )
                    exprs_file <- paste0("data/", exprs_file_basename, ".txt")
                    if (file.exists(pdata_file) && file.exists(exprs_file)) {
                        if (!exists("pdata")) {
                            cat(" Loading:", pdata_file_basename, "\n")
                            pdata <- read.delim(pdata_file, row.names=1)
                            rownames(pdata) <- make.names(rownames(pdata))
                            pdata <- cbind(pdata, rep(
                                which(dataset_name == all_dataset_names)[1],
                                nrow(pdata)
                            ))
                            colnames(pdata)[ncol(pdata)] <- "Batch"
                            pdata <- pdata[!is.na(pdata$Class), ]
                        }
                        eset_name <- paste0(
                            c("eset", exprs_file_basename), collapse="_"
                        )
                        cat("Creating:", eset_name, "\n")
                        exprs <- read.delim(exprs_file, row.names=NULL)
                        rowGroup <- as.vector(exprs[, 1])
                        exprs[, 1] <- NULL
                        if (any(duplicated(rowGroup))) {
                            exprs <- collapseRows(
                                exprs, rowGroup, row.names(exprs),
                                method=args$gex_collapse_meth
                            )$datETcollapsed
                        } else {
                            row.names(exprs) <- rowGroup
                        }
                        if (nrow(pdata) > ncol(exprs)) {
                            pdata <- pdata[colnames(exprs), , drop=FALSE]
                        } else if (nrow(pdata) < ncol(exprs)) {
                            exprs <- exprs[, rownames(pdata), drop=FALSE]
                        }
                        if (dataset_name %in% rna_seq_dataset_names) {
                            # filter low counts first before log transform
                            # exprs <- exprs[rowSums(exprs) > 1, ]
                            # exprs <- exprs[apply(exprs, 1, max) > 1, ]
                            # exprs <- exprs[-caret::nearZeroVar(t(exprs)), ]
                            exprs <- log2(exprs + 1)
                        } else if (dataset_name %in% needs_log2_dataset_names) {
                            exprs <- log2(exprs + 1)
                        }
                        eset <- ExpressionSet(
                            assayData=as.matrix(exprs),
                            phenoData=AnnotatedDataFrame(pdata)
                        )
                        assign(eset_name, eset)
                        save(
                            list=eset_name,
                            file=paste0("data/", eset_name, ".Rda")
                        )
                    }
                }
            }
        }
        if (exists("pdata")) remove(pdata)
    }
}
