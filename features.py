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
	params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':20,
		'shuffle':True,'random_state':1337,'class_weight':None}
	clf = SGDClassifier(**params)
	#clf = LogisticRegression(C=2.3,class_weight='auto')
	#clf = BernoulliNB(alpha=0.03)
	fio.encode(feats)
	train, truth = fio.transformTrain(feats)
	c = classifier.Classifier(train,truth)
	return c.holdout(clf,nFolds=10,fraction=0.2,seed=213)

def MultiGreedy():
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

def MultiGreedyReduction():
	pool = Pool(processes=3)
	fio = fileio.Preprocessed('../data/allQuadsFractions.csv',
			train='../data/train.csv',
			test='../data/test.csv')

	lastScore = 0
	bestFeatures = [3, 4, 17, 10, 123, 129, 292, 32, 9, 2, 7, 76, 5, 429, 663, 427, 308, 594, 13, 22, 279, 107, 360, 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
	while True:
		testFeatureSets = []
		for f in bestFeatures:
			testFeatureSets.append([x for x in bestFeatures if x != f])
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
	pool.close()
	print bestFeatures

def Predict():
	params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':30,
		'shuffle':True,'random_state':1337,'class_weight':None}
	clf = SGDClassifier(**params)
	#clf = LogisticRegression(C=3)
	#clf = BernoulliNB(alpha=0.0003)
	#fio = fileio.Preprocessed('../data/tripsFractions.csv')
	fio = fileio.RawInput('../data/alldata.csv',usePairs=True)
	#base = [127, 96, 53, 3, 103, 71, 151, 1, 65, 152]
	#base = [98,336,294,19,205,290,226,211,244,38,9,208,18,35,148,295,341,262,12,210, 233, 338, 0, 320]
	#base = [98,336,294,19,205,290,226,211,244,38,9,18,35,148,295,341,262,12,210, 233, 0, 320] # seed 1337
	#base = [212,94, 47, 291, 81, 121, 205, 204, 295, 138, 7, 258, 210, 234, 282, 0, 320] # seed 918
	#base = [176, 127, 154, 63, 289, 209, 15, 226, 300, 205]
	#base = [7, 204, 14, 96, 264, 294, 176, 127, 154, 63, 289, 209, 15, 226, 300, 205]
	#base = [240, 38, 48, 18, 212, 63, 12, 205, 263, 65, 262, 0, 338, 122, 300, 98, 210, 295, 320] # SGD
	#base = [289, 332, 201, 260, 235, 240, 38, 48, 18, 212, 63, 12, 205, 263, 65, 262, 0, 338, 122, 300, 98, 210, 295, 320]
	#base = [256, 302, 142, 243, 289, 341, 294, 104, 313, 135, 235, 204, 216, 38, 46, 332, 65, 268, 117, 207, 68, 208, 122, 0, 338, 318, 300, 308, 210, 295, 317]  #seed 213
	#base = [313, 291, 151, 64, 67, 20, 290, 112, 155, 138, 18, 285, 66, 212, 233, 204, 7, 208, 68, 282, 0, 210, 9, 295, 317] # seed 622
	#base = [201, 294, 260, 67, 220, 235, 7, 176, 290, 48, 309, 156, 66, 263, 138, 262, 35, 18, 233, 208, 240, 338, 0, 210, 9, 295, 317] # seed 410
	#base = [73, 8, 13, 68, 56, 11, 61, 36, 57, 34, 33, 66, 84, 1, 80, 2]
	base = [f for f in xrange(fio.df.shape[1]) if f != 8]
	#for b in base:
	#	print "%d. %s" %(b,fio.df.columns[b])
	#return
	fio.encode(base)
	train, truth = fio.transformTrain(base)
	c = classifier.Classifier(train, truth)
	prefix = 'lib/sgdPairs'
	c.validate(clf,nFolds=10,out=prefix+'.csv')
	#score = c.holdout(clf,nFolds=10,fraction=0.2)
	#print score

	if True:
		test = fio.transformTest(base)
		print test.shape
		clf.fit(train,truth)
		y_ = clf.predict_proba(test)[:,1]
		writeSubmission(y_,filename=prefix+'Test.csv')
		return


def HyperSearch():
	#base = [0, 10, 11, 20, 37, 38, 39,42, 43, 48, 54, 61, 62,64, 65, 68, 70, 72, 82, 83, 86]
	#base = [0, 7, 9, 10, 11, 32, 37, 42, 43,
	# 		48, 57, 60, 63, 64, 65, 66, 67, 68, 
	# 		70, 72, 80, 83, 86, 89]
	#base = [3, 4, 17, 10, 123, 129, 292, 32, 9, 2, 7, 76, 5, 429, 663, 427, 308, 594, 13, 22, 279, 107, 360, 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
	#base = [212,98,336,294,19,205,290,226,211,244,38,9,208,18,35,148,295,341,262,12,210, 233, 338, 0, 320]
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
