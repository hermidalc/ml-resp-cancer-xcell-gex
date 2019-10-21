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
    "nci_sb_act_skcm",
    "ovarian",
    "radiation",
    "schalper",
    "tcga_skcm",
    "vanallen_sd0",
    "vanallen_sd1",
    "zhao"
)
count_dataset_names <- c(
    "gse78220",
    "gse91061_sd0",
    "gse91061_sd1",
    "gse91061_pre_sd0",
    "gse91061_pre_sd1",
    "gse91061_on_sd0",
    "gse91061_on_sd1",
    "gse106291",
    "gse115821",
    "gse121810",
    "miao",
    "nci_sb_act_skcm",
    "ovarian",
    "tcga_skcm",
    "vanallen_sd0",
    "vanallen_sd1",
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
    "gex_cyt",
    "gex_cyt_tmb",
    "gex_tmb",
    "tmb",
    "xcell"
)
meta_types <- c(
    "sd0",
    "sd1"
)
norm_methods <- c(
    "none",
    "counts",
    "fpkm",
    "rpkm",
    "tmm",
    "tpm",
    "vst"
)
feat_types <- c(
    "none",
    "ensembl",
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
    "cyt",
    "go0002376",
    "ifng",
    "ifnge",
    "impres",
    "ipres",
    "kdm5a",
    "kdm5a2",
    "kdm5a3",
    "kdm5a4",
    "kdm5a_cyt",
    "kdm5a_cyt_tmb",
    "kdm5a_tmb",
    "kleg",
    "kleg_cyt",
    "kleg_cyt_tmb",
    "kleg_tmb",
    "pdl1"
)
common_pheno_names <- c(
    "Batch",
    "Class"
)
pdata_cls_name <- "Class"
pdata_bat_name <- "Batch"
pdata_grp_name <- "Group"
wes_tmb_feat_name <- "WES-TMB"
gex_cyt_feat_name <- "GEX-CYT"
matfact_k <- 20
