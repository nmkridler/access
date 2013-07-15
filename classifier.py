""" classifier.py
"""
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cross_validation import KFold, train_test_split, StratifiedKFold
from plotting import PlotROC, PlotDensity
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedShuffleSplit
# Author: Nick Kridler

class Classifier(object):
	"""
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def validate(self,clf,nFolds=10,out='out.csv'):
		""""""
		kf = StratifiedKFold(self.y,n_folds=nFolds,indices=False)
		#kf = KFold(len(self.y),n_folds=nFolds,indices=False,shuffle=True,random_state=1337)
		y_ = np.empty(len(self.y))
		mean_auc = 0.
		rows = np.arange(self.X.shape[0])
		for train, test in kf:
			train_, test_ = rows[train], rows[test]
			clf.fit(self.X[train_,:],self.y[train_])
			y_[test_] = clf.predict_proba(self.X[test_])[:,1]
			fpr, tpr, thresholds = roc_curve(self.y[test_], y_[test_])
			thisAuc = auc(fpr,tpr)
			print "AUC: %f" % thisAuc
			mean_auc += thisAuc

		print mean_auc/len(kf)
		PlotROC(self.y,y_,printAuc=True)
		pl.show()
		pl.figure()
		PlotDensity(y_[self.y==0],'H0',minval=0,maxval=1)
		PlotDensity(y_[self.y==1],'H1',minval=0,maxval=1)
		pl.show()
		np.savetxt(out,y_,delimiter=',')

	def stratifiedHoldout(self,clf,nFolds=20,fraction=0.3,seed=1337):
		""""""
		meanAuc = 0.
		sss = StratifiedShuffleSplit(self.y,n_iter=nFolds,test_size=fraction)
		for train, test in sss:
			clf.fit(self.X[train,:],self.y[train])
			y_ = clf.predict_proba(self.X[test,:])[:,1]
			fpr, tpr, threshold = roc_curve(self.y[test],y_)
			rocAuc = auc(fpr,tpr)
			meanAuc += rocAuc
		return meanAuc/nFolds

	def holdout(self,clf,nFolds=20,fraction=0.3,seed=1337):
		""""""
		meanAuc = 0.
		#pl.figure()
		for i in xrange(nFolds):
			xTrain, xCV, yTrain, yCV = train_test_split(self.X,
														self.y,
														test_size=fraction,
														random_state=i*seed)

			clf.fit(xTrain,yTrain)
			y_ = clf.predict_proba(xCV)[:,1]

			fpr, tpr, threshold = roc_curve(yCV,y_)
			rocAuc = auc(fpr,tpr)

			#print "AUC (fold %d/%d): %f" %(i+1,nFolds,rocAuc)
			meanAuc += rocAuc
			#PlotROC(yCV,y_)
		#pl.show()

		#print "Mean AUC: %f" % (meanAuc/nFolds)
		return meanAuc/nFolds

