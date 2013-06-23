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
from sklearn.ensemble import RandomForestClassifier
def main():

	f = fileio.Fileio('../data/alldata.csv')
	train, truth = f.transformTrain('../data/train.csv')

	c = classifier.Classifier(train, truth)
	clf = LogisticRegression(C=3)
	clf = SGDClassifier(loss='log',penalty='l2',alpha=0.0001,n_iter=30,shuffle=True,random_state=1337)
	c.validate(clf,nFolds=10,out='sgdBase30.csv')
	print clf.coef_.shape
	pl.figure()
	pl.semilogy(np.sort(np.abs(clf.coef_[0.,:]))[::-1],np.arange(1,clf.coef_.shape[1]+1),lw=3)
	pl.show()
	x = np.abs(c.coef_)
	sort_ = np.argsort(x)[::-1]

#	clf = SGDClassifier(loss='log',penalty='l2',alpha=0.0001,n_iter=30,shuffle=True,random_state=1337)
	clf = LogisticRegression(C=3)
	c.validate(clf,indices=sort_[:int(0.5*sort_.size)],nFolds=10,out='sgd20Pct30.csv',coef=False)
	print int(0.5*sort_.size)
	return
	test = f.transformTest('../data/test.csv')
	print test.shape
	clf.fit(train[:,sort_[:int(0.05*sort_.size)]],truth)
	y_ = clf.predict_proba(test[:,sort_[:int(0.05*sort_.size)]])
	print y_[:,1].min(), y_[:,1].max()
	out = open('623ReducedFix.sub','w')
	out.write('id,ACTION\n')
	for i in xrange(y_.shape[0]):
		out.write('%i,%f\n'%(i+1,y_[i,1]))
	out.close()
if __name__=="__main__":
	main()