import sys
sys.path.insert(1,'D:/nkridler/ybfu/python/')

import pylab as pl
import numpy as np
import pandas as pd
import classifier
import fileio
reload(fileio)
reload(classifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
def main():

	f = fileio.Fileio('../data/alldata.csv')
	train, truth = f.transformTrain('../data/train.csv')
	print train.shape
	c = classifier.Classifier(train, truth)
	clf = LogisticRegression(C=3)
	#clf = SGDClassifier(loss='log',penalty='l2',alpha=0.0001,n_iter=20,shuffle=True)
	#c.validate(clf,nFolds=4,out='logr.csv')
	#pl.show()
	#return
	test = f.transformTest('../data/test.csv')
	print test.shape
	clf.fit(train,truth)
	y_ = clf.predict_proba(test)
	print y_[:,1].min(), y_[:,1].max()
	out = open('612.sub','w')
	out.write('id,ACTION\n')
	for i in xrange(y_.shape[0]):
		out.write('%i,%f\n'%(i+1,y_[i,1]))
	out.close()
if __name__=="__main__":
	main()