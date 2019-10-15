select.fast.filter <- function(matrix, disc.method, threshold, attrs.nominal) {

    CalcGain <- function(m1, m2, symm) {
        dd <- length(m1)
        fq1 <- table(m1)
        fq1 <- fq1/dd[1]
        entropyF1 <- -sapply(fq1, function(z) if (z == 0) 0 else z * log(z))
        entropyF1 <- sum(entropyF1)
        fq2 <- table(m2)
        fq2 <- fq2/dd[1]
        entropyF2 <- -sapply(fq2, function(z) if (z == 0) 0 else z * log(z))
        entropyF2 <- sum(entropyF2)
        fq <- table(m1, m2)
        entropyF12 <- 0
        for (i in seq_len(length(fq2))) {
            fq0 <- fq[, i]/sum(fq[, i])
            vrem <- -sapply(fq0, function(z) if (z == 0) 0 else z * log(z))
            entropyF12 <- entropyF12 + (fq2[i]) * sum(vrem)
        }
        entropy <- entropyF1 - entropyF12
        if (symm) {
            if ((entropyF1 + entropyF2) == 0) {
                entropy <- 0
            } else {
                entropy <- 2 * entropy / (entropyF1 + entropyF2)
            }
        }
        return(entropy)
    }

    ProcessData <- function(matrix, disc.method, attrs.nominal, flag=FALSE) {
        dd <- dim(matrix)
        matrix <- data.frame(matrix)
        matrix[, dd[2]] <- as.factor(matrix[, dd[2]])
        if (disc.method == "MDL") {
            m3 <- FSelectorRcpp::discretize(
                as.formula(paste(names(matrix)[dd[2]], "~.")), matrix
            )
            # m3<-mdlp(matrix)$Disc.data
        }
        if (disc.method == "equal frequency") {
            m3 <- matrix
            for (i in 1:(dd[2] - 1)) {
                if (!(i %in% attrs.nominal)) {
                    m3[, i] <- arules::discretize(matrix[, i], breaks=3)
                }
            }
        }
        if (disc.method == "equal interval width") {
            m3 <- matrix
            for (i in 1:(dd[2] - 1)) {
                if (!(i %in% attrs.nominal)) {
                    m3[, i] <- arules::discretize(
                        matrix[, i], breaks=3, method="interval"
                    )
                }
            }
        }
        sel.feature <- 1:dd[2]
        if (flag) {
            # extract the features with one interval
            sel.one <- lapply(m3, function(z) {
                (length(levels(z)) == 1) && (levels(z) == "'All'")
            })
            sel.one <- which(unlist(sel.one) == TRUE)
            # selected features
            if (length(sel.one) > 0) {
                sel.feature <- sel.feature[-sel.one]
                matrix <- matrix[, -sel.one, drop=FALSE]
                m3 <- m3[, -sel.one, drop=FALSE]
            }
        }
        return(list(m3=m3, sel.feature=sel.feature))
    }

    out <- ProcessData(matrix, disc.method, attrs.nominal, flag=FALSE)
    m3 <- out$m3
    sel.feature <- out$sel.feature
    # algorithm
    dd <- dim(m3)
    if (dd[2] > 1) {
        # SU1=information.gain(names(matrix)[dd[2]]~., matrix)
        # entropy of feature
        entropy <- c()
        class <- m3[, dd[2]]
        for (j in 1:(dd[2] - 1)) {
            feature <- m3[, j]
            out <- CalcGain(feature, class, FALSE)
            entropy <- c(entropy, out)
        }
        ind <- sapply(entropy, function(z) z >= threshold)
        entropy <- entropy[ind]
        m3 <- m3[, ind, drop=FALSE]
        index.F1 <- 1
        res <- sort(entropy, decreasing=TRUE, index.return=TRUE)
        val <- res$ix
        entropy.sort <- res$x
        while (index.F1 <= length(val)) {
            Fp <- m3[, val[index.F1]]
            index.F2 <- index.F1 + 1
            while (index.F2 <= length(val)) {
                Fq <- m3[, val[index.F2]]
                SUpq <- CalcGain(Fp, Fq, FALSE)
                if (SUpq >= entropy.sort[index.F2]) {
                  val <- val[-index.F2]
                  entropy.sort <- entropy.sort[-index.F2]
                  index.F2 <- index.F2 - 1
                }
                index.F2 <- index.F2 + 1
            }
            index.F1 <- index.F1 + 1
        }
        # what features are selected, ind-features with SU(p,c)>threshold
        num.feature <- sel.feature[ind]
        num.feature <- num.feature[val]  #val - sorting
        info <- data.frame(names(m3)[val], entropy.sort, num.feature)
    } else {
        info <- data.frame(character(), numeric(), numeric())
    }
    names(info) <- c("Biomarker", "Information.Gain", "NumberFeature")
    return(info)
}
