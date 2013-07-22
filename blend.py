import pylab as pl
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy import optimize

def getData(fname,baseDir='/home/nick/amazon/access/'):
	files = pd.read_csv(fname)
	X = np.loadtxt(baseDir+files.train.iloc[0])
	for i in files.train.ix[1:]:
		X = np.vstack((X,np.loadtxt(baseDir+i)))
	train = X.transpose()

	X = np.array(pd.read_csv(baseDir+files.test.iloc[0]).ACTION)
	for i in files.test.ix[1:]:
		X = np.vstack((X,np.array(pd.read_csv(baseDir+i).ACTION)))
	test = X.transpose()
	print files.train.iloc[cols]
	return train, test

def fopt_pred(pars,data):
	return np.dot(data, pars)

def fopt(pars):
	fpr, tpr, thresholds = metrics.roc_curve(y,fopt_pred(pars,Xtrain))
	return -metrics.auc(fpr,tpr)


cols = range(5)
trainD, testD = getData('subList.csv')
y = np.array(pd.read_csv('../data/train.csv').ACTION)
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
Xtrain = scl.fit_transform(trainD[:,cols])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=2.3,class_weight='auto')
#clf = Ridge(alpha=40000.)
import classifier
c = classifier.Classifier(Xtrain,y)
c.validate(clf,nFolds=10,out='blendLogit.csv')
#n = len(cols)
#x0 = clf.coef_
#x0 /= np.sum(x0)
#xopt = optimize.minimize(fopt,x0,method='Nelder-Mead')

#print -fopt(xopt.x)
#print xopt.x
#print -fopt(x0)
#print x0
#out = pd.read_csv('./subs/nb_predict.csv')
#out.ACTION = fopt_pred(xopt.x,testD[:,cols])
#out.to_csv('blend5.csv',index=False)
