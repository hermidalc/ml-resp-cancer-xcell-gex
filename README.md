# ML Prediction of Drug Response in Cancer Using xCell and Gene Expression Data

Using machine learning methods to predict drug response (immunotherapy, chemo, etc) in cancer patients.

1. Install Conda Dependencies

```bash
conda install unzip
```

3. Install xCell

```R
source("https://bioconductor.org/biocLite.R")
biocLite("GSVA", suppressUpdates=TRUE)
options(unzip="internal")
library(devtools)
install_github("dviraran/xCell")
```

4. Dump xCell Genes

```R
library(xCell)
data(xCell.data)
write.table(sort(xCell.data$genes), file="xcell_symbols.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
