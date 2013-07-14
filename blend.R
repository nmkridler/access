base <- cbind(read.csv('/home/nick/amazon/data/train.csv')$ACTION,
          read.csv('/home/nick/amazon/access/subs/logQuads704.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/rf234Train.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/rfRawTrain.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/log708Train.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/log709Train.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/log701.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/gbmWRawTrain.csv',header=F),
          read.csv('/home/nick/amazon/access/subs/gbm234Train.csv',header=F),      
          read.csv('/home/nick/amazon/access/subs/nbTrain.csv',header=F)  )

test <- cbind(read.csv('/home/nick/amazon/access/subs/704LogQuads.csv'),
               read.csv('/home/nick/amazon/access/subs/rf234Test.csv')$ACTION,
               read.csv('/home/nick/amazon/access/subs/rf234Test.csv')$ACTION,
               read.csv('/home/nick/amazon/access/subs/708logTest.csv')$ACTION,
               read.csv('/home/nick/amazon/access/subs/709logTest.csv')$ACTION,
               read.csv('/home/nick/amazon/access/subs/701LogTrips.csv')$ACTION,
               read.csv('/home/nick/amazon/access/subs/gbmWRawTest.csv')$ACTION,  
               read.csv('/home/nick/amazon/access/subs/gbm234Test.csv')$ACTION,   
               read.csv('/home/nick/amazon/access/subs/nb_predict.csv')$ACTION)
colnames(base)[1] <- 'ACTION'
cols <- c(5,6,7,9)
library("Metrics")
fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m)
}
fn.opt <- function(pars) {
  -auc(train$ACTION, fn.opt.pred(pars, train[,cols]))
}

nTrials <- 100
frac <- as.integer(0.7*dim(base)[1])
coef <- array(0,c(nTrials,length(cols)))
aucVals <- array(0,nTrials)
bestAuc <- array(0,c(nTrials,(dim(base)[2]-1)))
train <- base
pars <- rep(1/length(cols),length(cols))
opt.result <- optim(pars, fn.opt,control = list(trace = T))
all.pred <- fn.opt.pred(opt.result$par, train[,cols])
allAuc <- array(0,nTrials)
saveCoef <- opt.result$par
for(i in 1:nTrials){
  trainRows <- sample(dim(base)[1],frac)
  train <- base[trainRows,]
  pars <- rep(1/length(cols),length(cols))
  opt.result <- optim(pars, fn.opt,control = list(trace = T))
  coef[i,] <- opt.result$par
  test.pred <- fn.opt.pred(opt.result$par, base[-trainRows,cols])
  aucVals[i] <- auc(base$ACTION[-trainRows],test.pred)
  allAuc[i] <- auc(base$ACTION[-trainRows],all.pred[-trainRows])
  for(j in 2:dim(base)[2]){
    bestAuc[i,j-1] <- auc(base$ACTION[-trainRows],base[-trainRows,j]) 
  }
}
colors <- c('red','blue','green','orange','purple','cyan')
plot(density(aucVals),lwd=3)
lines(density(allAuc),lwd=3,col='purple')
for(i in 1:4){
  print(c("AUC: ",auc(base$ACTION,base[,cols[i]])))
  print(c("Mean AUC: ", mean(bestAuc[,cols[i]-1])))
  lines(density(bestAuc[,cols[i]-1]),lwd=3,col=colors[i])
}

coefAvg <- colMeans(coef)
train.pred <- fn.opt.pred(coefAvg, base[,cols])
for( i in cols ){
  print(auc(base$ACTION,base[,i]))
}

print(auc(base$ACTION,train.pred))
print(coefAvg)
print(saveCoef)
test.pred <- fn.opt.pred(opt.result$par, test[,cols])
out <- test[,1:2]
colnames(out) <- c('Id','ACTION')
out$ACTION <- test.pred
#write.csv(out,file='/home/nick/amazon/access/optimblend3.csv',row.names=F)