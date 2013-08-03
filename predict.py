import pylab as pl
import numpy as np
import pandas as pd
import plotting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc

import classifier
reload(classifier)
def main():
	makeSub = True
	featureImportance = False
	cvfold = True
	df = pd.read_csv('../data/cprobTrain15NA.csv')

	X, y = np.array(pd.read_csv('../data/train.csv',usecols=range(1,9))), np.array(pd.read_csv('../data/train.csv').ACTION)
	X = np.hstack((X,np.array(df)))

	params = {'max_depth':4, 'subsample':0.5, 'verbose':0, 'random_state':1337,
		'min_samples_split':10, 'min_samples_leaf':10, 'max_features':10,
		'n_estimators': 350, 'learning_rate': 0.05}	

	clf = GradientBoostingClassifier(**params)
	prefix = 'lib/gbm350d4m10c15'
	if cvfold:
		c = classifier.Classifier(X,y)
		c.validate(clf,nFolds=10,out=prefix+'Train.csv')

	if makeSub:
		Xt = np.array(pd.read_csv('../data/test.csv',usecols=range(1,9)))
		Xt = np.hstack((Xt,np.array(pd.read_csv('../data/cprobTest15NA.csv'))))
		clf.fit(X,y)
		y_ = clf.predict_proba(Xt)[:,1]
		out = pd.read_csv('subs/nbBaseTest.csv')
		out.ACTION = y_
		out.to_csv(prefix+'Test.csv',index=False)

	if featureImportance:
		print "Feature ranking:"
		importances = clf.feature_importances_
		indices = np.argsort(importances)[::-1]
		np.savetxt('indices.txt',indices,delimiter=',')
		for f in xrange(df.shape[1]):
			print "%d. feature (%s,%f)" % (f + 1, df.columns[indices[f]], importances[indices[f]])



	
if __name__=="__main__":
	main()