import numpy as np
import pylab as pl
import pandas as pd
import scipy.sparse as sp

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

def buildQuads(x):
	"""Category Triplets"""
	pairs = []
	for i in xrange(len(x)):
		for j in xrange(i+1,len(x)): 
			for k in xrange(j+1,len(x)):
				for l in xrange(k+1,len(x)):
					pairs.append([x[i],x[j],x[k],x[l]])	
	return pairs

def sortAndMerge(df,key,truth,threshold=25):
	""" Sort by column counts and merge to data frame"""
	# Sort the unique values by counts
	yHX = df.ix[:NUMTRAIN,key].value_counts().order()[::-1]
	yH1 = df.ix[truth == 1,key].value_counts().order()[::-1]
	yAll = yHX[yH1.index].values.astype('float')
	cP = yH1.values/yAll
	if np.sum(yAll > threshold) == 0:
		return df
	medVal = np.median(cP[yAll > threshold])
	cP[yAll < threshold] = medVal
	cpDF = pd.DataFrame({'id':yH1.index,'cp':cP,'nulls':1*np.array(yAll < threshold)})

	y = df[key].value_counts().order()[::-1]
	allDF = pd.merge(cpDF,pd.DataFrame({'id':y.index,'counts':y}),
		how='right',on='id',sort=False)
	allDF.cp = allDF.cp.fillna(medVal)
	allDF.nulls = allDF.nulls.fillna(1)

	suffix = 'Ids'
	df = pd.merge(df,
		pd.DataFrame({key:allDF.id,key+suffix:allDF.cp,key+'.nulls':allDF.nulls}),
		how='inner',on=key,sort=False)
	return df

def add4grams(df,pairs,truth):
	""" Turn trigrams into unique columns """
	keys = [ col for col in df.columns]
	df = addIDColumn(df)
	for pair in pairs:
		# Make a key using the column names
		key = '%s.%s.%s.%s'%(pair[0],pair[1],pair[2],pair[3])
		keys.append(key)

		# Create a new column containing the combo
		df[key] = map(lambda x: '%i.%i.%i.%i'%(x[0],x[1],x[2],x[3]),
			zip(df[pair[0]],df[pair[1]],df[pair[2]],df[pair[3]]))

		df = sortAndMerge(df,key,truth)

	return df.ix[:,[c for c in df.columns if c not in keys]]

def addTrigrams(df,pairs,truth):
	""" Turn trigrams into unique columns """
	keys = [ col for col in df.columns]
	df = addIDColumn(df)
	for pair in pairs:
		# Make a key using the column names
		key = '%s.%s.%s'%(pair[0],pair[1],pair[2])
		keys.append(key)

		# Create a new column containing the combo
		df[key] = map(lambda x: '%i.%i.%i'%(x[0],x[1],x[2]),
			zip(df[pair[0]],df[pair[1]],df[pair[2]]))

		df = sortAndMerge(df,key,truth)

	return df.ix[:,[c for c in df.columns if c not in keys]]


def addBigrams(df,pairs,truth):
	""" Turn bigrams into unique columns """
	keys = [ col for col in df.columns]
	df = addIDColumn(df)
	for pair in pairs:
		# Make a key using the column names
		key = '%s.%s'%(pair[0],pair[1])
		keys.append(key)

		# Create a new column containing the combo
		df[key] = map(lambda x: '%i.%i'%(x[0],x[1]),
			zip(df[pair[0]],df[pair[1]]))

		df = sortAndMerge(df,key,truth)

	return df.ix[:,[c for c in df.columns if c not in keys]]

def threshold(df, truth):
	""" Add the ability to threshold """
	h = df.columns
	df = addIDColumn(df)
	for c in h:
		df = sortAndMerge(df,c,truth)

	df = df.ix[:,[c for c in df.columns if c not in h]]
	return df

class CondProb(object):
	""" Fileio helper """
	def __init__(self, train='../data/alldata.csv',truth='../data/train.csv',
			usePairs=False, useTrips=False, useQuads=False):

		data = pd.read_csv(train)
		self.truth = np.array(pd.read_csv(truth).ACTION)
		self.df = threshold(data.copy(),self.truth) 

		if usePairs:
			pairs = buildPairs(data.columns)
			self.df = pd.merge(self.df,addBigrams(data.copy(),pairs,self.truth),how='inner',on='ID')

		if useTrips:
			trips = buildTrips(data.columns)
			self.df = pd.merge(self.df,addTrigrams(data.copy(),trips,self.truth),how='inner',on='ID')

		if useQuads:
			quads = buildQuads(data.columns)
			self.df = pd.merge(self.df,add4grams(data.copy(),quads,self.truth),how='inner',on='ID')

		self.df['type'] = self.df.ID.apply(lambda x: x.split('.')[0])
		self.df['order'] = self.df.ID.apply(lambda x: int(x.split('.')[1]))
		trainDf = self.df.ix[self.df.type == 'train',:].sort(columns=['order'])
		testDf = self.df.ix[self.df.type == 'test',:].sort(columns=['order'])
		ignore = ['type','order','ID']
		cols = [c for c in trainDf.columns if c not in ignore]
		trainDf.ix[:,cols].to_csv('cprobTrain.csv',index=False)
		testDf.ix[:,cols].to_csv('cprobTest.csv',index=False)
