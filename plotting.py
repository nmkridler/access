""" plotting.py
"""

# Author: Nick Kridler
from sklearn.metrics import roc_curve, auc
import pylab as pl

def binaryHistogram(df,x,y,xlim=[],params={},labels={}):
	""" Plots a histogram of series

		Plots the histogram of a DataFrame column 
		conditioned on a binary labeled column.

		Args:
			df: pandas DataFrame
			x: column to histogram
			y: column containing binary condition
			xlim: x-bounds of plot
			params: dictionary containing pylab.hist parameters
			labels: dictionary containing axes labels

	"""
	pl.figure()
	df[x].ix[df[y] == 0].hist(color='black',**params)
	df[x].ix[df[y] == 1].hist(color='red',**params)
	pl.xlim(xlim)
	pl.title(labels['title'])
	pl.xlabel(labels['xlabel'])
	pl.ylabel(labels['ylabel'])
	pl.show()


def PlotROC(truth, prediction):
	"""Plot a roc curve"""
	fpr, tpr, thresholds = roc_curve(truth, prediction)

	roc_auc = auc(fpr,tpr)
	print "Area under the curve: %f" % roc_auc
	pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=3)
	pl.ylim([0.0, 1.0])
	pl.xlim([0.0, 1.0])
	pl.xlabel('PFA')
	pl.ylabel('PD')
	pl.legend(loc="lower right")
	return			