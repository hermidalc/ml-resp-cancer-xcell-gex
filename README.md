# ML Prediction of Drug Response in Cancer Using xCell and Gene Expression Data

Using machine learning methods to predict drug response (immunotherapy, chemo, etc) in cancer patients.

1. Install System Dependencies

```bash
sudo dnf install -y unzip
```
2. Install Conda Dependencies

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda install -v -y \
bioconductor-gsva \
bioconductor-gseabase
```

3. Install xCell

```R
options(unzip="internal")
library(devtools)
install_github("dviraran/xCell")
```

4. Dump xCell Genes

```R
library(xCell)
data(xCell.data)
write.table(sort(xCell.data$genes), file="xCell_symbols.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
