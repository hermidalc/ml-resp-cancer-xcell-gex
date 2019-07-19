# ML Prediction of Drug Response in Cancer Using xCell and Gene Expression Data

Using machine learning methods to predict drug response (immunotherapy, chemo, etc) in cancer patients.

1. Install xCell and Dump Genes

```R
options(unzip="internal")
library(devtools)
install_github("dviraran/xCell")
library(xCell)
data(xCell.data)
write.table(sort(xCell.data$genes), file="xcell_symbols.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
```
