library("Metrics")
LOADDATA <- T
MAXMODELS <- 15

if (LOADDATA) {
  source("~/amazon/access/loadData.R")
  
  # Load the training data
  data <- loadData('~/amazon/access/lib/sublist.csv','~/amazon/access/lib/')
  train <- data$train
  test  <- data$test
  
  # Find the column with the max auc
  nModels <- dim(data$train)[2]
}

fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m)
}

fn.opt <- function(pars) {
  -auc(train$ACTION, fn.opt.pred(pars, train[,cols]))
}

maxAuc <- 0.
maxLoc <- 2
for (i in 2:nModels) {
  iAuc <- auc(train$ACTION,train[,i])
  if (iAuc > maxAuc) {
    maxAuc <- iAuc
    maxLoc <- i
  }
}
maxCols <- maxLoc
allCols <- 2:nModels

# Loop until max models
saveAuc <- array(0,MAXMODELS)
saveLoc <- array(1,MAXMODELS)
saveAuc[1] <- maxAuc
saveLoc[1] <- maxLoc
for (iter in 2:MAXMODELS) {
  maxAuc <- 0.
  maxLoc <- 0.
  for (i in setdiff(allCols,maxCols)) {
    cols <- c(maxCols,i)
    pars <- rep(1/length(cols),length(cols))
    opt.result <- optim(pars, fn.opt,control = list(trace = T))
    iAuc <- -fn.opt(opt.result$par)
    if (iAuc > maxAuc) {
      saveAuc[iter] <- iAuc
      saveLoc[iter] <- i
      maxAuc <- iAuc
    }
  }
  maxCols <- c(maxCols,saveLoc[iter])
}
print(maxCols)
print(saveAuc)
plot(saveAuc)


