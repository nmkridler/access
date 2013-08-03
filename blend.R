# Courtesy of Lucas aka Leustagos on Kaggle

library("Metrics")
LOADDATA <- T

if (LOADDATA) {
  source("~/amazon/access/loadData.R")
  
  # Load the training data
  data <- loadData('~/amazon/access/lib/sublist.csv')
  base <- data$train
  test  <- data$test
  
  train <- data$train
  
  # Find the column with the max auc
  nModels <- dim(data$train)[2]
}

cols <- c(6:9,10,24,39,41)

fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m)
}
fn.opt <- function(pars) {
  -auc(train$ACTION, fn.opt.pred(pars, train[,cols]))
}

# Full prediction
pars <- rep(1/length(cols),length(cols))
opt.result <- optim(pars, fn.opt,control = list(trace = T,maxit=1500))
all.pred <- fn.opt.pred(opt.result$par, train[,cols])
full.pred <- fn.opt.pred(opt.result$par, base[,cols])
print(opt.result$par)

# Loop over trials
nTrials <- 500
frac <- as.integer(0.7*dim(train)[1])
allAuc <- array(0,nTrials)
bestAuc <- array(0,c(nTrials,dim(train)[2]))
for(i in 1:nTrials){
  trainRows <- sample(dim(train)[1],frac)
  allAuc[i] <- auc(train$ACTION[-trainRows],all.pred[-trainRows])
  for(j in 2:dim(train)[2]){
    bestAuc[i,j-1] <- auc(train$ACTION[-trainRows],train[-trainRows,j]) 
  }
}

colors <- c('red','blue','green','orange','purple','cyan')
plot(density(allAuc),lwd=3,col='black',xlim=c(0.85,0.94))
for(i in 1:length(cols)){
  print(c("AUC: ",auc(train$ACTION,train[,cols[i]])))
  print(c("Mean AUC: ", mean(bestAuc[,cols[i]-1])))
  lines(density(bestAuc[,cols[i]-1]),lwd=3,col=colors[i])
}


for( i in cols ){
  print(colnames(train)[i])
  print(auc(train$ACTION,train[,i]))
}
print(auc(train$ACTION,all.pred))
print(auc(base$ACTION,full.pred))
write.csv(all.pred,file='/Users/nkridler/amazon/access/lib/trainOptim.csv',row.names=F)
test.pred <- fn.opt.pred(opt.result$par, test[,cols])
out <- test[,1:2]
colnames(out) <- c('Id','ACTION')
out$ACTION <- test.pred
write.csv(out,file='/Users/nkridler/amazon/access/optimblend.csv',row.names=F)