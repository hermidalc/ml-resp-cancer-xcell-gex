#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--datasets", type="character", nargs="+", help="datasets")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument('--gex-collapse-meth', type="character", default="MaxMean", help="gex collapse rows method")
args <- parser$parse_args()
all_dataset_names <- dataset_names
if (!is.null(args$datasets)) {
    dataset_names <- intersect(dataset_names, args$datasets)
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
        pdata_file_basename <- paste0(c(dataset_name, suffixes, "meta"), collapse="_")
        pdata_file <- paste0("data/", pdata_file_basename, ".txt")
        if (data_type %in% c("wxs", "xcell")) {
            exprs_file_basename <- paste0(c(dataset_name, suffixes), collapse="_")
            exprs_file <- paste0("data/", exprs_file_basename, ".txt")
            if (file.exists(pdata_file) && file.exists(exprs_file)) {
                cat(" Loading:", pdata_file_basename, "\n")
                pdata <- read.delim(pdata_file, row.names=1)
                rownames(pdata) <- make.names(rownames(pdata))
                pdata <- cbind(pdata, rep(which(dataset_name == all_dataset_names)[1], nrow(pdata)))
                colnames(pdata)[ncol(pdata)] <- "Batch"
                eset_name <- paste0(c("eset", exprs_file_basename), collapse="_")
                cat("Creating:", eset_name, "\n")
                exprs <- read.delim(exprs_file, row.names=1)
                eset <- ExpressionSet(
                    assayData=as.matrix(exprs),
                    phenoData=AnnotatedDataFrame(pdata)
                )
                eset <- eset[, !is.na(eset$Class)]
                assign(eset_name, eset)
                save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
            }
        }
        else if (data_type == "gex") {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (!(suffix %in% c("none", "None"))) suffixes <- c(suffixes, suffix)
                    }
                    exprs_file_basename <- paste0(c(dataset_name, suffixes), collapse="_")
                    exprs_file <- paste0("data/", exprs_file_basename, ".txt")
                    if (file.exists(pdata_file) && file.exists(exprs_file)) {
                        if (!exists("pdata")) {
                            cat(" Loading:", pdata_file_basename, "\n")
                            pdata <- read.delim(pdata_file, row.names=1)
                            rownames(pdata) <- make.names(rownames(pdata))
                            pdata <- cbind(pdata, rep(which(dataset_name == all_dataset_names)[1], nrow(pdata)))
                            colnames(pdata)[ncol(pdata)] <- "Batch"
                        }
                        eset_name <- paste0(c("eset", exprs_file_basename), collapse="_")
                        cat("Creating:", eset_name, "\n")
                        exprs <- read.delim(exprs_file, row.names=NULL)
                        rowGroup <- as.vector(exprs[,1])
                        exprs[,1] <- NULL
                        if (dataset_name %in% needs_log2_dataset_names) {
                            exprs <- log2(exprs + 1)
                        }
                        if (any(duplicated(rowGroup))) {
                            exprs <- data.frame(WGCNA::collapseRows(
                                exprs, rowGroup, row.names(exprs), method=args$gex_collapse_meth
                            )$datETcollapsed)
                        }
                        else {
                            row.names(exprs) <- rowGroup
                        }
                        eset <- ExpressionSet(
                            assayData=as.matrix(exprs),
                            phenoData=AnnotatedDataFrame(pdata)
                        )
                        eset <- eset[, !is.na(eset$Class)]
                        assign(eset_name, eset)
                        save(list=eset_name, file=paste0("data/", eset_name, ".Rda"))
                    }
                }
            }
        }
        if (exists("pdata")) remove(pdata)
    }
}
