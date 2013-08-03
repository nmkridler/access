loadData <- function(filename,libDir,labels='D:/amazon/data/train.csv') {
  files <- read.csv(filename)
  train <- read.csv(labels)$ACTION
  for( file in files$train){
    train <- cbind(train,read.csv(paste(libDir,file,sep=""),header=F))
  }
  colnames(train) <- c('ACTION',levels(files$train))
  test <- read.csv(paste(libDir,files$test[1],sep=""))
  for( file in files$test[-1]){
    test <- cbind(test,read.csv(paste(libDir,file,sep=""))$ACTION)
  }
  colnames(test) <- c('Id',levels(files$test))
  list( train=train,
        test=test)
}