""" classifier.py
"""
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cross_validation import KFold, train_test_split
from plotting import PlotROC
from sklearn.metrics import roc_curve, auc
# Author: Nick Kridler

class Classifier(object):
	"""
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y

	def validate(self,clf,nFolds=10,out='out.csv'):
		""""""
		kf = KFold(len(self.y),n_folds=nFolds,indices=False,shuffle=True,random_state=1337)
		y_ = np.empty((len(self.y),2))
		mean_auc = 0.
		rows = np.arange(self.X.shape[0])
		for train, test in kf:
			train_, test_ = rows[train], rows[test]
			clf.fit(self.X[train_,:],self.y[train_])
			y_[test_,:] = clf.predict_proba(self.X[test_])
			fpr, tpr, thresholds = roc_curve(self.y[test_], y_[test_,1])
			thisAuc = auc(fpr,tpr)
			print "AUC: %f" % thisAuc
			mean_auc += thisAuc

		print mean_auc/len(kf) 
		PlotROC(self.y,y_[:,1])

		np.savetxt(out,y_[:,1],delimiter=',')

	def holdout(self,clf,nFolds=20,fraction=0.3,seed=1337):
		""""""
		meanAuc = 0.
		pl.figure()
		for i in xrange(nFolds):
			xTrain, xCV, yTrain, yCV = train_test_split(self.X,
														self.y,
														test_size=fraction,
														random_state=i*seed)

			clf.fit(xTrain,yTrain)
			y_ = clf.predict_proba(xCV)[:,1]

			fpr, tpr, threshold = roc_curve(yCV,y_)
			rocAuc = auc(fpr,tpr)

			print "AUC (fold %d/%d): %f" %(i+1,nFolds,rocAuc)
			meanAuc += rocAuc
			PlotROC(yCV,y_)
		pl.show()

		print "Mean AUC: %f" % (meanAuc/nFolds)


