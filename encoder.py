""" encoder.py: transform 
"""
# Author: Nick Kridler

import numpy as np
import pandas as pd

def ThresholdEncoder(df,x,threshold=25):
	""" Turn categorical variables into binary features """
	uniq = df[x].value_counts().order()[::-1]
	count = np.sum(uniq >= threshold)

	map_ = {}
	for i in xrange(count):
		catStr = x+'Cat%06i'%i
		df[catStr] = 1*(df[x] == uniq.index[i])
		map_[uniq.index[i]] = catStr

	df[x+'CatLow'] = 0
	for i in xrange(count,uniq.index.size):
		df[x+'CatLow'] += 1*(df[x] == uniq.index[i])
		map_[uniq.index[i]] = x+'CatLow'

	return map_

