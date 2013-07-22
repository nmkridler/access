library("Metrics")
LOADDATA <- T

if (LOADDATA) {
  source("D:/amazon/access/loadData.R")
  
  # Load the training data
  data <- loadData('D:/amazon/access/lib/sublist.csv','D:/amazon/access/lib/')
  train <- data$train
  test  <- data$test
  
  # Find the column with the max auc
  nModels <- dim(data$train)[2]
}
#[1]  8 11 23  5 15  7  3 22  9 13 24 19 18  6 16
#[1] 0.9110791 0.9125963 0.9135320 0.9136529 0.9139294 0.9131478 0.9142116 0.9138964 0.9133700 0.9129934 0.9132347
#[12] 0.9130404 0.9131356 0.9132225 0.9131728

#cols <- c(5,10,7,6)
cols <- c(8,11,23,5,15)
fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m)
}
fn.opt <- function(pars) {
  -auc(train$ACTION, fn.opt.pred(pars, train[,cols]))
}

# Full prediction
pars <- rep(1/length(cols),length(cols))
opt.result <- optim(pars, fn.opt,control = list(trace = T))
all.pred <- fn.opt.pred(opt.result$par, train[,cols])
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

test.pred <- fn.opt.pred(opt.result$par, test[,cols])
out <- test[,1:2]
colnames(out) <- c('Id','ACTION')
out$ACTION <- test.pred
#write.csv(out,file='/home/nick/amazon/access/optimblendthresh.csv',row.names=F)