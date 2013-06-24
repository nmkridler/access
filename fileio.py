import numpy as np
import pylab as pl
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

NUMTRAIN = 32769
NUMTEST = 58921

def addIDColumn(df):
	"""Add an id column for joining later"""
	df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]),
		zip(['train']*NUMTRAIN + ['test']*NUMTEST, range(NUMTRAIN) + range(NUMTEST)))
	return df

def buildPairs(x):
	"""Category Pairs"""
	pairs = []
	for i in xrange(len(x)):
		for j in xrange(i+1,len(x)): 
			pairs.append([x[i],x[j]])	
	return pairs

def buildTrips(x):
	"""Category Triplets"""
	pairs = []
	for i in xrange(len(x)):
		for j in xrange(i+1,len(x)): 
			for k in xrange(j+1,len(x)):
				pairs.append([x[i],x[j],x[k]])	
	return pairs

def sortAndMerge(df,key,fraction=0.42):
	""" Sort by column counts and merge to data frame"""
	# Sort the unique values by counts
	y = df[key].value_counts().order()[::-1]

	# Take the top N percent
	counts = int(y.size*fraction)
	index = np.arange(y.size,dtype='int32')
	index[counts:] = counts

	df = pd.merge(df,
		pd.DataFrame({key:y.index,key+'Ids':index}),
		how='inner',on=key,sort=False)
	return df

def addTrigrams(df,pairs,fraction=0.42):
	""" Turn trigrams into unique columns """
	keys = [ col for col in df.columns]
	df = addIDColumn(df)
	for pair in pairs:
		# Make a key using the column names
		key = '%s.%s.%s'%(pair[0],pair[1],pair[2])
		keys.append(key)

		# Create a new column containing the combo
		df[key] = map(lambda x: '%i,%i,%i'%(x[0],x[1],x[2]),
			zip(df[pair[0]],df[pair[1]],df[pair[2]]))

		df = sortAndMerge(df,key,fraction=fraction)

	return df.ix[:,[c for c in df.columns if c not in keys]]


def addBigrams(df,pairs,fraction=0.5):
	""" Turn bigrams into unique columns """
	keys = [ col for col in df.columns]
	df = addIDColumn(df)
	for pair in pairs:
		# Make a key using the column names
		key = '%s.%s'%(pair[0],pair[1])
		keys.append(key)

		# Create a new column containing the combo
		df[key] = map(lambda x: '%i,%i'%(x[0],x[1]),
			zip(df[pair[0]],df[pair[1]]))

		df = sortAndMerge(df,key,fraction=fraction)

	return df.ix[:,[c for c in df.columns if c not in keys]]

def threshold(df, fraction=1.0):
	""" Add the ability to threshold """
	h = df.columns
	col = dict([(y+'Ids',y) for y in df.columns])
	df = addIDColumn(df)
	if fraction == 1.0:
		return df 
	
	for c in h:
		df = sortAndMerge(df,c,fraction=fraction)

	df = df.ix[:,[c for c in df.columns if c not in h]]
	df = df.rename(columns=col)
	return df

class Fileio(object):
	""" Fileio helper """
	def __init__(self,filename='',usePairs=False,useTrips=False):
		# Read all the data
		df = pd.read_csv(filename)

		self.df = threshold(df.copy(),fraction=1.0)

		if usePairs:
			pairs = buildPairs(df.columns)
			self.df = pd.merge(self.df,addBigrams(df.copy(),pairs,fraction=1.0),how='inner',on='ID')

		if useTrips:
			trips = buildTrips(df.columns)
			self.df = pd.merge(self.df,addTrigrams(df.copy(),trips,fraction=0.42),how='inner',on='ID')

		# Create a OneHotEncoder
		self.encoder = OneHotEncoder()
		usecols = [c for c in self.df.columns if c != 'ID']
		features = [0, 8, 9, 10, 19, 34, 36, 37,
					38, 41, 42, 43, 47, 53, 55, 60,
					61, 63, 64, 67, 69, 71, 75, 81, 82, 85]
		features = [0, 8] + [f+1 for f in features[2:]]
		#usecols = [f+1 for f in features])
		#self.df = self.df.ix[:,features]
		#print self.df.columns

		usecols = [c for c in self.df.columns if c != 'ID']
		self.encoder.fit(np.array(self.df.ix[:,usecols],dtype='float'))

	def transformTrain(self,filename):
		""" Transform the training set"""
		df = pd.read_csv(filename,usecols=range(9))
		df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]), zip(['train']*NUMTRAIN, range(NUMTRAIN)))
	
		x = pd.merge(df.ix[:,['ID','ACTION']],self.df,how='left',on='ID',sort=False)
		ignore = ['ID','ACTION']
		usecols = [c for c in x.columns if c not in ignore]
		return self.encoder.transform(np.array(x.ix[:,usecols],dtype='float')), np.array(x.ACTION)

	def transformTest(self,filename):
		""" Transform the testing set"""
		df = pd.read_csv(filename)
		df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]), zip(['test']*NUMTEST, range(NUMTEST)))

		x = pd.merge(df.ix[:,['ID','ROLL_CODE']],self.df,how='left',on='ID',sort=False)
		ignore = ['ID','ROLL_CODE']
		usecols = [c for c in x.columns if c not in ignore]
		return self.encoder.transform(np.array(x.ix[:,usecols],dtype='float'))




