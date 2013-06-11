import sys
sys.path.insert(1,'D:/nkridler/ybfu/python/')

import pylab as pl
import numpy as np
import pandas as pd
import analytics
from models import Classifier

def main():
	c = Classifier(filename='../workspace/sjmcDSAmets.csv',label='label')
	from sklearn.ensemble import RandomForestClassifier
	params = {'n_estimators':50, 'min_samples_split':10, 'min_samples_leaf':10}
	clf = RandomForestClassifier(**params)
	c.validate(clf)
	pl.show()

if __name__=="__main__":
	main()