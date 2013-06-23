import pylab as pl
import numpy as np
import pandas as pd
import classifier
import fileio
import plotting
reload(fileio)
reload(classifier)
reload(plotting)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

def writeSubmission(y,filename='submission.csv'):
	out = open(filename,'w')
	out.write('id,ACTION\n')
	for i in xrange(y.size):
		out.write('%i,%f\n'%(i+1,y[i]))
	out.close()


def main():

	f = fileio.Fileio('../data/alldata.csv',useTrips=False)
	train, truth = f.transformTrain('../data/train.csv')

	c = classifier.Classifier(train, truth)
	#clf = LogisticRegression(C=3)
	params = {'loss':'log','penalty':'l2','alpha':0.0001,'n_iter':30,
		'shuffle':True,'random_state':1337}
	clf = SGDClassifier(**params)
	#c.validate(clf,nFolds=10,out='sgdBase30.csv')
	c.holdout(clf,nFolds=20,fraction=0.2)

	if False:
		test = f.transformTest('../data/test.csv')
		print test.shape
		clf.fit(train,truth)
		y_ = clf.predict_proba(test)[:,1]
		writeSubmission(y_,filename='out.csv')

	
if __name__=="__main__":
	main()