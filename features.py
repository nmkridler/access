import numpy as np
import pandas as pd

import classifier
import fileio
reload(fileio)
reload(classifier)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from multiprocessing import Pool
# 1st: 89: 0.899896
# 2nd: 63: 0.900192
# 3rd: 7: 0.900354
# remove 44: 0.901041
# remove 76: 0.901387
# remove 56: 0.901669
# remove 35: 0.901999
def writeSubmission(y,filename='submission.csv'):
	out = open(filename,'w')
	out.write('id,ACTION\n')
	for i in xrange(y.size):
		out.write('%i,%f\n'%(i+1,y[i]))
	out.close()

def featureScore(x):
	fio, feats = x
	#params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
	#	'shuffle':True,'random_state':1337,'class_weight':None}
	#clf = SGDClassifier(**params)
	clf = LogisticRegression(C=2.3,class_weight='auto')
	fio.encode(feats)
	train, truth = fio.transformTrain(feats)
	c = classifier.Classifier(train,truth)
	return c.holdout(clf,nFolds=10,fraction=0.2)

def MultiGreedy():
	pool = Pool(processes=2)
	fio = fileio.Preprocessed('../data/quadsFractions.csv',
			train='../data/train.csv',
			test='../data/test.csv')

	#bestFeatures = [92, 9, 53, 89, 40, 67, 7, 62, 11, 32, 65, 48, 19, 37, 86, 0, 68]
	bestFeatures = [98,336,294,19,205,290,226,211,244,38,9,208,18,35,148,295,341,262,12,210, 233, 338, 0, 320] #[59, 161, 48, 19, 37, 158, 0, 139]
	lastScore = 0
	allFeatures = [f for f in xrange(345,fio.df.shape[1]) if f != 8]
	while True:
		testFeatureSets = [[f] + bestFeatures for f in allFeatures if f not in bestFeatures]
		args = [(fio,fSet) for fSet in testFeatureSets]
		scores = pool.map(featureScore,args)
		(score, featureSet) = max(zip(scores,testFeatureSets))
		print featureSet
		print score
		if score <= lastScore:
			break
		lastScore = score
		bestFeatures = featureSet

	pool.close()
	print bestFeatures


def GreedyFeatures():
	#clf = LogisticRegression(C=2.3,class_weight='auto')
	params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None}
	clf = SGDClassifier(**params)
	fio = fileio.Preprocessed('../data/triplets.csv',
		train='../data/train.csv',
		test='../data/test.csv')
	#base = [0, 7, 9, 10, 11, 32, 37, 42, 43,
	# 		48, 57, 60, 63, 64, 65, 66, 67, 68, 
	# 		70, 72, 80, 83, 86, 89]

	print "Performing greedy feature selection..."
	base = []
	score_hist = []
	good_features = []
	last_score = 0
	# Greedy feature selection loop
	while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
		scores = []
		for iter_ in xrange(fio.df.shape[1]-1):
			if iter_ == 8:
				continue
			if iter_ not in good_features:
				feats = list(good_features) + [iter_]
				x = (fio,feats)
				score = featureScore(x)
				scores.append((score, iter_))
				print "Feature: %i Mean AUC: %f" % (iter_, score)
		good_features.add(sorted(scores)[-2][1])
		score_hist.append(sorted(scores)[-2])
		print "Current features: %s" % sorted(list(good_features))

	# Remove last added feature from good_features
	good_features.remove(score_hist[-1][1])
	good_features = sorted(list(good_features))
	print good_features

def Predict():
	params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None}
	#clf = SGDClassifier(**params)
	clf = LogisticRegression(C=2.3,class_weight='auto')
	fio = fileio.Preprocessed('../data/quadsPt25.csv')
	#fio = fileio.RawInput('../data/alldata.csv',usePairs=True,useTrips=True,useQuads=False)
	#fio.df.to_csv('../data/tripsFractions.csv',index=False)
	#return
	#base = [0, 10, 11, 20, 37, 38, 39,	
	#		42, 43, 48, 54, 61, 62,
	#		64, 65, 68, 70, 72, 82, 83, 86]
	#base = range(8) + range(9,f.df.shape[1]-1)
	#base = [32, 38, 72, 60, 10, 84, 54, 70, 9, 65, 14, 34, 64, 7, 59, 48, 89, 20, 37, 66, 0, 86, 19, 11, 68]
	#base = [70, 92, 9, 53, 89, 40, 67, 7, 62, 11, 32, 65, 48, 19, 37, 86, 0, 68]
	#base = [72, 59, 66, 10, 70, 92, 9, 53, 89, 40, 67, 7, 62, 11, 32, 65, 48, 19, 37, 86, 0, 68]
	#base = [98,336,294,19,205,290,226,211,244,38,9,208,18,35,148,295,341,262,12,210, 233, 338, 0, 320]
	#base = [67, 82, 130, 143, 32, 10, 60, 42, 98, 162, 48, 68, 128, 93, 86, 65, 11, 7, 64, 120, 69, 34, 84, 37, 70, 0]
	#base = [9,146,98,105, 66, 32, 10, 138, 99, 141, 42, 124, 34, 143, 103, 107, 144, 11, 7, 59, 161, 48, 19, 37, 158, 0, 139]
	#base = [0, 7, 9, 10, 11, 32, 37, 42, 43, 48, 57, 60, 63, 64, 65, 66, 67, 68, 70, 72, 80, 83, 86, 89]
	fio.encode(base)
	train, truth = fio.transformTrain(base)

	c = classifier.Classifier(train, truth)
	c.validate(clf,nFolds=10,out='log701quadsPt25.csv')
	score = c.holdout(clf,nFolds=10,fraction=0.2)
	print score

	if True:
		test = fio.transformTest(base)
		print test.shape
		clf.fit(train,truth)
		y_ = clf.predict_proba(test)[:,1]
		writeSubmission(y_,filename='701LogQuadsPt25.csv')
		return

def GreedyReduction():

	clf = LogisticRegression(C=2.3,class_weight='auto')
	f = fileio.Preprocessed('../data/triplets.csv')

	#base = [0, 10, 11, 20, 37, 38, 39,
	#		42, 43, 48, 54, 61, 62,
	#		64, 65, 68, 70, 72, 82, 83, 86]
	base = [0, 7, 9, 10, 11, 32, 37, 42, 43,
	 		48, 57, 60, 63, 64, 65, 66, 67, 68, 
	 		70, 72, 80, 83, 86, 89]
	print "Performing greedy feature reduction..."
	score_hist = []
	N = 10
	base = []
	good_features = set(base)
	# Greedy feature selection loop
	while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
		scores = []
		for iter_ in good_features:
			feats = [ii for ii in good_features if ii != iter_]
			f.encode(feats)
			train, truth = f.transformTrain('../data/train.csv',feats)
			c = classifier.Classifier(train, truth)
			score = c.holdout(clf,nFolds=10,fraction=0.2)
			scores.append((score, iter_))
			print "Feature: %i Mean AUC: %f" % (iter_, score)
		maxI = sorted(scores)[-1][1]
		good_features = [ii for ii in good_features if ii != maxI]
		score_hist.append(sorted(scores)[-1])
		print "Current features: %s" % sorted(list(good_features))

	# Remove last added feature from good_features
	good_features.remove(score_hist[-1][1])
	good_features = sorted(list(good_features))
	print good_features
	return

def HyperSearch():
	base = [0, 10, 11, 20, 37, 38, 39,
			42, 43, 48, 54, 61, 62,
			64, 65, 68, 70, 72, 82, 83, 86]
	#base = [0, 7, 9, 10, 11, 32, 37, 42, 43,
	# 		48, 57, 60, 63, 64, 65, 66, 67, 68, 
	# 		70, 72, 80, 83, 86, 89]
	f = fileio.Preprocessed('../data/triplets.csv')	
	f.encode(base)
	train, truth = f.transformTrain('../data/train.csv',base)
	print "Performing hyperparameter selection..."

	params = {'loss':'log','penalty':'elasticnet','alpha':0.0001,'n_iter':30,
		'shuffle':True,'random_state':1337,'class_weight':None}
	clf = SGDClassifier(**params)
	#clf = LogisticRegression(C=2.3,class_weight='auto')
	# Hyperparameter selection loop
	score_hist = []
	Cvals = np.logspace(-4, 4, 15, base=2)
	eval_ = classifier.Classifier(train, truth)
	for C in Cvals:
		clf.C = C
		score = eval_.holdout(clf,nFolds=10,fraction=0.2)
		score_hist.append((score,C))
		print "C: %f Mean AUC: %f" %(C, score)
	bestC = sorted(score_hist)[-1][1]
	print "Best C value: %f" % (bestC)

def main():
	#Predict()
	MultiGreedy()

if __name__ == "__main__":
	main()
