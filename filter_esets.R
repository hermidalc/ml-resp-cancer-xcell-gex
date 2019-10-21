#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))

proj_root_dir <- dirname(sys.frame(1)$ofile)
source(paste(proj_root_dir, "config.R", sep="/"))

parser <- ArgumentParser()
parser$add_argument(
    "--filter", type="character", nargs="+", help="filter function"
)
parser$add_argument(
    "--dataset-tr", type="character", nargs="+", help="dataset tr"
)
parser$add_argument(
    "--dataset-te", type="character", nargs="+", help="dataset te"
)
parser$add_argument(
    "--num-combo-tr", type="integer", help="num datasets to combine"
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
    "--prep-meth", type="character", nargs="+", help="preprocess method"
)
parser$add_argument(
    "--bc-meth", type="character", nargs="+", help="batch correction method"
)
parser$add_argument(
    "--filt-type", type="character", nargs="+", help="filter type"
)
parser$add_argument(
    "--load-only", action="store_true", default=FALSE,
    help="show search and eset load only"
)
args <- parser$parse_args()
if (
    is.null(args$filter) ||
    !(args$filter %in% c("common_features", "features"))
) {
    stop("--filter option required")
}
if (args$filter == "features") {
    if (is.null(args$num_combo_tr)) stop("--num-combo-tr option required")
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
if (!is.null(args$prep_meth)) {
    prep_methods <- intersect(prep_methods, args$prep_meth)
}
if (!is.null(args$bc_meth)) {
    bc_methods <- intersect(bc_methods, args$bc_meth)
}
if (!is.null(args$filt_type)) {
    filt_types <- intersect(filt_types, args$filt_type)
}
if (args$filter == "common_features") {
    if (!is.null(args$dataset_tr)) {
        dataset_names <- intersect(dataset_names, args$dataset_tr)
    }
    for (dataset_name in dataset_names) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    eset_name <- paste0(
                        c("eset", dataset_name, suffixes), collapse="_"
                    )
                    eset_file <- paste0("data/", eset_name, ".Rda")
                    if (file.exists(eset_file)) {
                        cat(" Loading:", eset_name, "\n")
                        load(eset_file)
                        # get common features
                        if (args$filter == "common_features") {
                            if (exists("common_feat_names")) {
                                common_feat_names <- intersect(
                                    common_feat_names,
                                    featureNames(get(eset_name))
                                )
                            } else {
                                common_feat_names <-
                                    featureNames(get(eset_name))
                            }
                        }
                    }
                }
            }
        }
    }
    if (args$load_only) quit()
    for (dataset_name in dataset_names) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    eset_name <- paste0(
                        c("eset", dataset_name, suffixes), collapse="_"
                    )
                    if (exists(eset_name)) {
                        eset_filt_name <- paste0(
                            c(eset_name, "cff"), collapse="_"
                        )
                        cat("Creating:", eset_filt_name, "\n")
                        eset_filt <- get(eset_name)
                        eset_filt <- eset_filt[common_feat_names, ]
                        assign(eset_filt_name, eset_filt)
                        save(
                            list=eset_filt_name,
                            file=paste0("data/", eset_filt_name, ".Rda")
                        )
                    }
                }
            }
        }
    }
} else if (args$filter == "features") {
    if (!is.null(args$dataset_tr) && !is.null(args$num_combo_tr)) {
        dataset_tr_name_combos <- combn(
            intersect(dataset_names, args$dataset_tr),
            as.integer(args$num_combo_tr)
        )
    } else if (!is.null(args$dataset_tr)) {
        dataset_tr_name_combos <- combn(
            intersect(dataset_names, args$dataset_tr),
            length(args$dataset_tr)
        )
    } else {
        dataset_tr_name_combos <- combn(
            dataset_names, as.integer(args$num_combo_tr)
        )
    }
    if (!is.null(args$dataset_te)) {
        dataset_te_names <- intersect(dataset_names, args$datasets_te)
    } else {
        dataset_te_names <- dataset_names
    }
    for (col in 1:ncol(dataset_tr_name_combos)) {
        for (data_type in data_types) {
            for (norm_meth in norm_methods) {
                for (feat_type in feat_types) {
                    suffixes <- c(data_type)
                    for (suffix in c(norm_meth, feat_type)) {
                        if (suffix != "none") suffixes <- c(suffixes, suffix)
                    }
                    for (prep_meth in prep_methods) {
                        for (bc_meth in bc_methods) {
                            suffixes_tr <- suffixes
                            suffixes_te <- suffixes
                            if (prep_meth != "none") {
                                suffixes_tr <- c(suffixes_tr, prep_meth)
                                if (bc_meth != "none") {
                                    suffixes_tr <- c(suffixes_tr, bc_meth)
                                }
                                if (prep_meth != "mrg") {
                                    suffixes_te <- suffixes_tr
                                } else if (bc_meth != "none") {
                                    suffixes_te <- c(suffixes_te, bc_meth)
                                }
                            } else if (bc_meth != "none") {
                                suffixes_tr <- c(suffixes_tr, bc_meth)
                                suffixes_te <- suffixes_tr
                            }
                            if (
                                length(dataset_tr_name_combos[, col]) > 1 ||
                                bc_meth != "none"
                            ) {
                                eset_tr_name <- paste0(c(
                                    "eset", dataset_tr_name_combos[, col],
                                    suffixes_tr, "tr"
                                ), collapse="_")
                            } else {
                                eset_tr_name <- paste0(c(
                                    "eset", dataset_tr_name_combos[, col],
                                    suffixes_tr
                                ), collapse="_")
                            }
                            eset_tr_file <- paste0(
                                "data/", eset_tr_name, ".Rda"
                            )
                            # special case: gex + tpm + symbol + cyt
                            if (
                                data_type == "gex" && norm_meth != "tpm" &&
                                feat_type == "symbol" && filt_types == c("cyt")
                            ) next
                            if (!exists(eset_tr_name)) {
                                if (file.exists(eset_tr_file)) {
                                    cat(" Loading:", eset_tr_name, "\n")
                                    load(eset_tr_file)
                                } else {
                                    next
                                }
                            }
                            if (args$load_only) next
                            for (filt_type in filt_types) {
                                filt_feats_name <- paste0(c(
                                    filt_type, paste0(feat_type, "s")
                                ), collapse="_")
                                if (!exists(filt_feats_name)) {
                                    filt_feats_file <- paste0(
                                        "data/", filt_feats_name, ".txt"
                                    )
                                    if (file.exists(filt_feats_file)) {
                                        cat(
                                            " Reading:",
                                            basename(filt_feats_file)
                                        )
                                        filt_feats <- unique(
                                            readLines(filt_feats_file)
                                        )
                                        filt_feats <- trimws(
                                            filt_feats, which="both"
                                        )
                                        filt_feats <-
                                            filt_feats[filt_feats != ""]
                                        cat(
                                            " (", length(filt_feats),
                                            " unique features)\n", sep=""
                                        )
                                        assign(filt_feats_name, filt_feats)
                                    } else {
                                        next
                                    }
                                }
                                if (
                                    length(dataset_tr_name_combos[, col]) > 1 ||
                                    bc_meth != "none"
                                ) {
                                    eset_tr_filt_name <- paste0(c(
                                        "eset", dataset_tr_name_combos[, col],
                                        suffixes_tr, filt_type, "tr"
                                    ), collapse="_")
                                } else {
                                    eset_tr_filt_name <- paste0(c(
                                        eset_tr_name, filt_type
                                    ), collapse="_")
                                }
                                common_feat_names <- intersect(
                                    featureNames(get(eset_tr_name)),
                                    get(filt_feats_name)
                                )
                                # special case: gex + tpm + symbol + cyt
                                if (
                                    data_type == "gex" &&
                                    feat_type == "symbol" &&
                                    filt_type == "cyt"
                                ) {
                                    if (norm_meth == "tpm") {
                                        cat(
                                            "Creating:", eset_tr_filt_name,
                                            "\n"
                                        )
                                        cat(
                                            "  Filter:",
                                            length(common_feat_names), "/",
                                            length(get(filt_feats_name)),
                                            "features\n"
                                        )
                                        eset_tr_filt <- ExpressionSet(
                                            assayData=matrix(
                                                colMeans(exprs(
                                                    get(eset_tr_name)
                                                )[common_feat_names, ]),
                                                nrow=1,
                                                dimnames=list(
                                                    c(gex_cyt_feat_name),
                                                    colnames(exprs(
                                                        get(eset_tr_name)
                                                    ))
                                                )
                                            ),
                                            phenoData=phenoData(
                                                get(eset_tr_name)
                                            )
                                        )
                                    } else {
                                        next
                                    }
                                } else {
                                    cat("Creating:", eset_tr_filt_name, "\n")
                                    cat(
                                        "  Filter:",
                                        length(common_feat_names), "/",
                                        length(get(filt_feats_name)),
                                        "features\n"
                                    )
                                    eset_tr_filt <- get(eset_tr_name)
                                    eset_tr_filt <-
                                        eset_tr_filt[common_feat_names, ]
                                }
                                assign(eset_tr_filt_name, eset_tr_filt)
                                save(list=eset_tr_filt_name, file=paste0(
                                    "data/", eset_tr_filt_name, ".Rda"
                                ))
                                remove(list=c(eset_tr_filt_name))
                                for (dataset_te_name in setdiff(
                                    dataset_te_names,
                                    dataset_tr_name_combos[, col]
                                )) {
                                    if (
                                        length(dataset_tr_name_combos[, col])
                                        > 1 || bc_meth != "none"
                                    ) {
                                        eset_te_name <- paste0(c(
                                            eset_tr_name, dataset_te_name, "te"
                                        ), collapse="_")
                                        eset_te_filt_name <- paste0(c(
                                            eset_tr_filt_name, dataset_te_name,
                                            "te"
                                        ), collapse="_")
                                    } else {
                                        # eset_te_name <- paste0(c(
                                        #     "eset", dataset_te_name,
                                        #     suffixes_te
                                        # ), collapse="_")
                                        # eset_te_filt_name <- paste0(c(
                                        #     eset_te_name, filt_type
                                        # ), collapse="_")
                                        next
                                    }
                                    eset_te_file <- paste0(
                                        "data/", eset_te_name, ".Rda"
                                    )
                                    if (!exists(eset_te_name)) {
                                        if (file.exists(eset_te_file)) {
                                            cat(" Loading:", eset_te_name, "\n")
                                            load(eset_te_file)
                                        } else {
                                            next
                                        }
                                        if (args$load_only) next
                                        common_feat_names <- intersect(
                                            featureNames(get(eset_te_name)),
                                            get(filt_feats_name)
                                        )
                                        # special case: gex + tpm + symbol + cyt
                                        if (
                                            data_type == "gex" &&
                                            feat_type == "symbol" &&
                                            filt_type == "cyt"
                                        ) {
                                            if (norm_meth == "tpm") {
                                                cat(
                                                    "Creating:",
                                                    eset_te_filt_name, "\n"
                                                )
                                                cat(
                                                    "  Filter:",
                                                    length(common_feat_names),
                                                    "/",
                                                    length(
                                                        get(filt_feats_name)
                                                    ),
                                                    "features\n"
                                                )
                                                eset_te_filt <- ExpressionSet(
                                                    assayData=matrix(
                                                        colMeans(exprs(
                                                            get(eset_te_name)
                                                        )[common_feat_names, ]),
                                                        nrow=1,
                                                        dimnames=list(
                                                            c(gex_cyt_feat_name),
                                                            colnames(exprs(get(
                                                                eset_te_name
                                                            )))
                                                        )
                                                    ),
                                                    phenoData=phenoData(
                                                        get(eset_te_name)
                                                    )
                                                )
                                            } else {
                                                next
                                            }
                                        } else {
                                            cat(
                                                "Creating:", eset_te_filt_name,
                                                "\n"
                                            )
                                            cat(
                                                "  Filter:",
                                                length(common_feat_names),
                                                "/",
                                                length(
                                                    get(filt_feats_name)
                                                ),
                                                "features\n"
                                            )
                                            eset_te_filt <- get(eset_te_name)
                                            eset_te_filt <- eset_te_filt[
                                                common_feat_names,
                                            ]
                                        }
                                        assign(eset_te_filt_name, eset_te_filt)
                                        save(
                                            list=eset_te_filt_name,
                                            file=paste0(
                                                "data/", eset_te_filt_name,
                                                ".Rda"
                                            )
                                        )
                                        remove(list=c(eset_te_filt_name))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
