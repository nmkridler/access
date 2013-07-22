import pylab as pl
import numpy as np
import pandas as pd
import plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

import classifier
reload(classifier)
def main():

	X, y = np.array(pd.read_csv('cprobTrain.csv')), np.array(pd.read_csv('../data/train.csv').ACTION)
	#X, y = np.array(pd.read_csv('../data/train.csv',usecols=range(1,9))), np.array(pd.read_csv('../data/train.csv').ACTION)
	#X = np.hstack((X,np.array(pd.read_csv('../data/cprobQuadsTrain.csv'))))
	#X = np.hstack((X,np.array(pd.read_csv('cprobTrain.csv'))))

	params = {'max_depth':8, 'subsample':0.5, 'verbose':0, 'random_state':1337,
		'min_samples_split':5, 'min_samples_leaf':5, 'max_features':5,
		'n_estimators': 350, 'learning_rate': 0.05}	
	#params = {'n_estimators':500,'min_samples_split':25,'min_samples_leaf':25,'verbose':True,'n_jobs':4}
	#clf = RandomForestClassifier(**params)
	clf = GradientBoostingClassifier(**params)
	c = classifier.Classifier(X,y)
	c.validate(clf,nFolds=10,out='subs/gbmTrain.csv')
	Xt = np.array(pd.read_csv('cprobTest.csv'))
	#Xt = np.array(pd.read_csv('../data/test.csv',usecols=range(1,9)))
	#Xt = np.hstack((Xt,np.array(pd.read_csv('../data/cprobQuadsTest.csv'))))
	clf.fit(X,y)
	y_ = clf.predict_proba(Xt)[:,1]

	out = pd.read_csv('subs/nbBaseTest.csv')
	out.ACTION = y_
	out.to_csv('subs/gbmTest.csv',index=False)

	
if __name__=="__main__":
	main()