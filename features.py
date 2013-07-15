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
	return c.stratifiedHoldout(clf,nFolds=10,fraction=0.2)

def MultiGreedy():
	pool = Pool(processes=4)
	fio = fileio.Preprocessed('../data/quads10Threshold.csv',
			train='../data/train.csv',
			test='../data/test.csv')

	lastScore = 0
	bestFeatures = []
	allFeatures = range(1,fio.df.shape[1])
	while True:
		testFeatureSets = [[f] + bestFeatures for f in allFeatures if f not in bestFeatures]
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
		'shuffle':True,'random_state':1337,'class_weight':'auto'}
	#clf = SGDClassifier(**params)
	clf = LogisticRegression(C=1.8,class_weight='auto')
	fio = fileio.Preprocessed('../data/quads10Threshold.csv')
	base = [127, 96, 53, 3, 103, 71, 151, 1, 65, 152]
	#params = {'n_estimators':50,'min_samples_split':5,'min_samples_leaf':5,'verbose':True,'n_jobs':4}
	#clf = RandomForestClassifier(**params)	
	#fio = fileio.RawInput('../data/alldata.csv',usePairs=True,useTrips=True,useQuads=True)
	#fio.df.to_csv('../data/quads25Threshold.csv',index=False)
	#return
	#base = [315, 344, 293, 798, 310, 739, 547, 511, 794,105,500, 709, 122, 74, 7, 362, 28, 596, 737,845, 546, 748, 0, 706, 618, 37, 799, 600]
	#base = [123, 129, 292, 32, 9, 2, 7, 76, 5,429, 663,427,308,594, 13, 22, 279, 107, 360, 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
	#base = [3, 4, 17, 10, 123, 129, 292, 32, 9, 2, 7, 76, 5, 429, 663, 427, 308, 594, 13, 22, 279, 107, 360, 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
	#base = [ 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
	#base = [3, 4, 17, 10, 123, 129, 292, 32, 2, 7, 76, 5, 429, 663, 427, 308, 594, 13, 22, 279, 107, 360, 15, 557, 16, 431, 23, 19, 724, 27]
	#base = [0, 10, 11, 20, 37, 38, 39,	42, 43, 48, 54, 61, 62, 64, 65, 68, 70, 72, 82, 83, 86]
	#base = range(8) + range(9,f.df.shape[1]-1)
	#base = [32, 38, 72, 60, 10, 84, 54, 70, 9, 65, 14, 34, 64, 7, 59, 48, 89, 20, 37, 66, 0, 86, 19, 11, 68]
	#base = [70, 92, 9, 53, 89, 40, 67, 7, 62, 11, 32, 65, 48, 19, 37, 86, 0, 68]
	#base = [72, 59, 66, 10, 70, 92, 9, 53, 89, 40, 67, 7, 62, 11, 32, 65, 48, 19, 37, 86, 0, 68]
	#base = [183, 266, 211, 178, 70, 363, 329, 248, 331, 327, 368,365,66,184, 7, 261, 234, 47, 322, 216, 310,177,127,9,38,18,262,313,343,180,35,362,267,174,254,12,182,205,382,0,292]
	#base = [35,362,267,174,254,12,182,205,382,0,292]
	#base = [212,98,336,294,19,205,290,226,211,244,38,9,208,18,35,148,295,341,262,12,210, 233, 338, 0, 320]
	#base = [ x for x in xrange(fio.df.shape[1]) if x != 8]
	#base = [67, 82, 130, 143, 32, 10, 60, 42, 98, 162, 48, 68, 128, 93, 86, 65, 11, 7, 64, 120, 69, 34, 84, 37, 70, 0]
	#base = [9,146,98,105, 66, 32, 10, 138, 99, 141, 42, 124, 34, 143, 103, 107, 144, 11, 7, 59, 161, 48, 19, 37, 158, 0, 139]
	#base = [0, 7, 9, 10, 11, 32, 37, 42, 43, 48, 57, 60, 63, 64, 65, 66, 67, 68, 70, 72, 80, 83, 86, 89]
	#base = range(1,fio.df.shape[1])
	#from sklearn.decomposition import NMF
	#nmf = NMF(n_components=40)
	fio.encode(base)
	train, truth = fio.transformTrain(base)
	#train = train.todense()
	#train = nmf.fit_transform(train)
	c = classifier.Classifier(train, truth)
	c.validate(clf,nFolds=10,out='log10ThreshTrain.csv')
	score = c.holdout(clf,nFolds=10,fraction=0.2)
	print score

	if True:
		test = fio.transformTest(base)
		print test.shape
		clf.fit(train,truth)
		y_ = clf.predict_proba(test)[:,1]
		writeSubmission(y_,filename='log10ThresTest.csv')
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
