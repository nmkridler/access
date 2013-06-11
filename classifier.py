""" classifier.py
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from analytics.plotting import PlotROC

# Author: Nick Kridler

class Classifier(object):
	"""
	"""
	def __init__(self,filename='',label=''):
		data = pd.read_csv(filename)
		data = data.fillna(0.)
		notLabel = [c for c in data.columns if c != label]
		self.X, self.y = np.array(data.ix[:,notLabel]), np.array(data[label])
		print "Positive Samples: %i" % np.sum(self.y)
		print "Negative Samples: %i" % len(self.y) - np.sum(self.y)

	def validate(self,clf,nFolds=2):
		""""""
		kf = KFold(len(self.y),n_folds=nFolds,indices=False,shuffle=True)
		y_ = np.empty((len(self.y),2))
		for train, test in kf:
			clf.fit(self.X[train,:],self.y[train])
			y_[test,:] = clf.predict_proba(self.X[test])

		PlotROC(self.y,y_[:,1])


