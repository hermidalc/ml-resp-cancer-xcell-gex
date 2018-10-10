# config
dataset_names <- c(
    "gse78220",
    "gse91061_pre_sd0",
    "gse91061_pre_sd1",
    "gse91061_on_sd0",
    "gse91061_on_sd1",
    "gse115821",
    "vanallen",
    "tcga_skcm",
    "chen",
    "gse93157",
    "radiation",
    "ovarian",
    "gse103668",
    "gse106291",
    "gse106977",
    "gse22513",
    "gse25055",
    "gse28796",
    "gse32603",
    "gse35935",
    "gse37645",
    "gse45670",
    "gse5462",
    "gse60331",
    "gse6434",
    "gse76360",
    "gse93375"
)
rna_seq_dataset_names <- c(
    "gse78220",
    "gse91061_pre_sd0",
    "gse91061_pre_sd1",
    "gse91061_on_sd0",
    "gse91061_on_sd1",
    "gse115821",
    "vanallen",
    "tcga_skcm",
    "ovarian",
    "gse106291"
)
needs_log2_dataset_names <- c(
    "gse37645",
    "gse45670",
    "gse5462",
    "gse6434"
)
data_types <- c(
    "gex",
    "xcell"
)
norm_methods <- c(
    "none",
    "pkm"
)
feat_types <- c(
    "none",
    "symbol"
)
prep_methods <- c(
    "none",
    "cff",
    "mrg"
)
bc_methods <- c(
    "none",
    "ctr",
    "std",
    "rta",
    "rtg",
    "qnorm",
    "cbt",
    "fab",
    "sva",
    "stica0",
    "stica025",
    "stica05",
    "stica1",
    "svd"
)
filt_types <- c(
    "none",
    "xcell",
    "go0002376",
    "kdm5a",
    "impres"
)
common_pheno_names <- c(
    "Batch",
    "Class"
)
matfact_k <- 20
