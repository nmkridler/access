""" classifier.py
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
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


