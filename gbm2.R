rm(list = ls(all = TRUE))

library(gbm)
set.seed(123)

#load data
training = read.csv("/home/nick/amazon/data/train.csv")
test = read.csv("/home/nick/amazon/data/test.csv")


#remove last column
training <- training[,1:(ncol(training)-1)]
#shuffle training data 
training <- training[sample.int(nrow(training)),]


#GBM model

#parameters
GBM_ITERATIONS = 5000
GBM_LEARNING_RATE = 0.01
GBM_DEPTH = 49
GBM_MINOBS = 10





#cross-validation to find the optimal number of trees
gbm1 <- gbm(ACTION~. ,
			distribution = "bernoulli",
			data = training,
			n.trees = GBM_ITERATIONS,
			interaction.depth = GBM_DEPTH,
			n.minobsinnode = GBM_MINOBS,
			shrinkage = GBM_LEARNING_RATE,
			bag.fraction = 0.5,
			train.fraction = 1.0,
			cv.folds=7,
			keep.data = FALSE,
			verbose = FALSE,
			class.stratify.cv=TRUE,
			n.cores = 3)

		
		
iterations_optimal <- gbm.perf(object = gbm1 ,plot.it = TRUE,oobag.curve = TRUE,overlay = TRUE,method="cv")
print(iterations_optimal)


rm(gbm1)

#GBM Fit
x <- training[,2:ncol(training)]
y <- training[,1]
gbm2 <- gbm.fit(x , y
			,distribution ="bernoulli"
			,n.trees = iterations_optimal
			,shrinkage = GBM_LEARNING_RATE
			,interaction.depth = GBM_DEPTH
			,n.minobsinnode = GBM_MINOBS
			,bag.fraction = 0.5
			,nTrain = nrow(training)
			,keep.data=FALSE
			,verbose = TRUE)
		

#save submission
test_id <- test[,1]
test_data <- test[,2:(ncol(test)-1)]
		
ACTION <- predict.gbm(object = gbm2, newdata=test_data, n.trees=iterations_optimal, type="response")

submit_file = cbind(test_id,ACTION)
write.table(submit_file, file="gbm_5.csv",row.names=FALSE, col.names=TRUE, sep=",")
