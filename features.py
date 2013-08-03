import numpy as np
import pandas as pd

import classifier
import fileio
reload(fileio)
reload(classifier)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
from sklearn.naive_bayes import BernoulliNB

def writeSubmission(y,filename='submission.csv'):
	out = open(filename,'w')
	out.write('id,ACTION\n')
	for i in xrange(y.size):
		out.write('%i,%f\n'%(i+1,y[i]))
	out.close()

def featureScore(x):
	fio, feats = x
	clf = LogisticRegression(C=2.3,class_weight='auto')
	fio.encode(feats)
	train, truth = fio.transformTrain(feats)
	c = classifier.Classifier(train,truth)
	return c.holdout(clf,nFolds=10,fraction=0.2,seed=213)

def MultiGreedy():
	# Courtesy of Miroslaw Horbal
	pool = Pool(processes=4)
	fio = fileio.Preprocessed('../data/tripsFractions.csv',
			train='../data/train.csv',
			test='../data/test.csv')

	lastScore = 0
	bestFeatures = []
	cols = [c.split('Ids')[0] for c in fio.df.columns]
	cMap = {}
	for i in xrange(len(cols)):
		if cMap.get(cols[i]) == None:
			cMap[cols[i]] = []
		cMap[cols[i]].append(i)
	allFeatures = [f for f in xrange(fio.df.shape[1]) if f != 8] #8 ID, or 0
	ignoreSet = []
	for feat in bestFeatures:
		ignoreSet += cMap[cols[feat]]	

	while True:
		testFeatureSets = [[f] + bestFeatures for f in allFeatures if f not in bestFeatures and f not in ignoreSet]
		if len(testFeatureSets) == 0:
			break
		args = [(fio,fSet) for fSet in testFeatureSets]
		scores = pool.map(featureScore,args)
		(score, featureSet) = max(zip(scores,testFeatureSets))
		print featureSet
		print score
		if score <= lastScore:
			break
		lastScore = score
		bestFeatures = featureSet
		ignoreSet = []
		for feat in featureSet:
			ignoreSet += cMap[cols[feat]]

	pool.close()
	print bestFeatures


def Predict():
	USERAW = False
	clf = LogisticRegression(C=2.3,class_weight='auto')
	if USERAW:
		fio = fileio.RawInput('../data/alldata.csv',usePairs=True,useTrips=True)
		fio.df.to_csv('../data/tripsFractions.csv',index=False)
	else:
		fio = fileio.Preprocessed('../data/tripsFractions.csv')

	base = [201, 294, 260, 67, 220, 235, 7, 176, 290, 48, 309, 156, 66, 263, 138, 262, 35, 18, 233, 208, 240, 338, 0, 210, 9, 295, 317] # seed 410
	for b in base:
		print "%d. %s" %(b,fio.df.columns[b])
	return
	fio.encode(base)
	train, truth = fio.transformTrain(base)
	c = classifier.Classifier(train, truth)
	prefix = 'lib/logr'
	c.validate(clf,nFolds=10,out=prefix+'.csv')
	score = c.holdout(clf,nFolds=10,fraction=0.2)
	print score

	if True:
		test = fio.transformTest(base)
		clf.fit(train,truth)
		y_ = clf.predict_proba(test)[:,1]
		writeSubmission(y_,filename=prefix+'Test.csv')
		return


def HyperSearch():
	# Courtesy of Miroslaw Horbal
	base = [127, 96, 53, 3, 103, 71, 151, 1, 65, 152]
	f = fileio.Preprocessed('../data/quads10Threshold.csv')	
	f.encode(base)
	train, truth = f.transformTrain(base)
	print "Performing hyperparameter selection..."

	clf = LogisticRegression(C=2.3,class_weight='auto')
	# Hyperparameter selection loop
	score_hist = []
	Cvals = np.linspace(1,4,32)
	eval_ = classifier.Classifier(train, truth)
	for C in Cvals:
		clf.C = C
		score = eval_.holdout(clf,nFolds=10,fraction=0.2)
		score_hist.append((score,C))
		print "C: %f Mean AUC: %f" %(C, score)
	bestC = sorted(score_hist)[-1][1]
	print "Best C value: %f" % (bestC)

def main():
	#HyperSearch()
	Predict()
	#MultiGreedy()#Reduction()

if __name__ == "__main__":
	main()
