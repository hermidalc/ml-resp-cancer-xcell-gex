# config
dataset_names <- c(
    "chen",
    "gse5462",
    "gse6434",
    "gse22513",
    "gse25055",
    "gse28796",
    "gse32603",
    "gse35935",
    "gse37645",
    "gse45670",
    "gse60331",
    "gse76360",
    "gse78220",
    "gse91061_sd0",
    "gse91061_sd1",
    "gse91061_pre_sd0",
    "gse91061_pre_sd1",
    "gse91061_on_sd0",
    "gse91061_on_sd1",
    "gse93157",
    "gse93375",
    "gse103668",
    "gse106291",
    "gse106977",
    "gse115821",
    "gse121810",
    "gse123728",
    "miao",
    "ovarian",
    "radiation",
    "schalper",
    "tcga_skcm",
    "vanallen",
    "zhao"
)
rna_seq_dataset_names <- c(
    "gse78220",
    "gse91061_sd0",
    "gse91061_sd1",
    "gse91061_on_sd0",
    "gse91061_on_sd1",
    "gse91061_pre_sd0",
    "gse91061_pre_sd1",
    "gse106291",
    "gse115821",
    "gse121810",
    "miao",
    "ovarian",
    "tcga_skcm",
    "vanallen",
    "zhao"
)
needs_log2_dataset_names <- c(
    "gse5462",
    "gse6434",
    "gse37645",
    "gse45670",
    "schalper"
)
data_types <- c(
    "gex",
    "wxs",
    "xcell"
)
norm_methods <- c(
    "none",
    "pkm",
    "tpm"
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
    "cd8",
    "go0002376",
    "ifng",
    "impres",
    "kdm5a",
    "kleg",
    "pdl1",
    "xcell"
)
common_pheno_names <- c(
    "Batch",
    "Class"
)
matfact_k <- 20
