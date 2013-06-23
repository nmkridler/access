import numpy as np
import pylab as pl
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder


def buildPairs(x):
	pairs = []
	for i in xrange(len(x)):
		for j in xrange(i+1,len(x)): 
			pairs.append([x[i],x[j]])	
	return pairs

def buildTrips(x):
	pairs = []
	for i in xrange(len(x)):
		for j in xrange(i+1,len(x)): 
			for k in xrange(j+1,len(x)):
				pairs.append([x[i],x[j],x[k]])	
	return pairs

def addTrigrams(df,pairs):
	h = df.columns
	df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]),
		zip(['train']*32769 + ['test']*58921, range(32769) + range(58921)))
	keys = [ col for col in h]
	total = 0
	for pair in pairs:
		key = '%s.%s.%s'%(pair[0],pair[1],pair[2])
		keys.append(key)
		df[key] = map(lambda x: '%i,%i,%i'%(x[0],x[1],x[2]),
			zip(df[pair[0]],df[pair[1]],df[pair[2]]))
		y = df[key].value_counts().order()[::-1]
		counts = int(y.size*0.42)
		total += counts
		index = np.arange(y.size,dtype='int32')
		index[counts:] = counts
		df = pd.merge(df,pd.DataFrame({key:y.index,key+'Ids':index}),how='inner',on=key,sort=False)
	print total
	return df.ix[:,[c for c in df.columns if c not in keys]]


def addBigrams(df,pairs):
	h = df.columns
	total = 0
	df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]),
		zip(['train']*32769 + ['test']*58921, range(32769) + range(58921)))
	keys = [ col for col in h]
	for pair in pairs:
		key = '%s.%s'%(pair[0],pair[1])
		keys.append(key)
		df[key] = map(lambda x: '%i,%i'%(x[0],x[1]),
			zip(df[pair[0]],df[pair[1]]))
		y = df[key].value_counts().order()[::-1]
		counts = int(np.sum(y >= 1000))
		if counts == 0:
			continue
		total += counts
		index = np.arange(y.size,dtype='int32')
		index[counts:] = counts
		df = pd.merge(df,pd.DataFrame({key:y.index,key+'Ids':index}),how='inner',on=key,sort=False)
	print total
	return df.ix[:,[c for c in df.columns if c not in keys]]

def threshold(df):
	h = df.columns
	col = dict([(y+'Ids',y) for y in h])
	df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]),
		zip(['train']*32769 + ['test']*58921, range(32769) + range(58921)))
	col['ID'] = 'ID'
	return df
	for c in h:
		y = df[c].value_counts().order()[::-1]
		counts = int(y.size*0.44)
		#counts = int(np.sum(y >= 30))
		index = np.arange(y.size,dtype='int32')
		index[counts:] = counts
		df = pd.merge(df,pd.DataFrame({c:y.index,c+'Ids':index}),how='inner',on=c,sort=False)
	df = df.ix[:,[c for c in df.columns if c not in h]]
	df = df.rename(columns=col)
	return df

class Fileio(object):
	def __init__(self,filename=''):
		df = pd.read_csv(filename)
		pairs = buildPairs(df.columns)
		trips = buildTrips(df.columns)
		self.header = df.columns
		self.df = threshold(df.copy())
		#print self.df.shape
		#self.df = pd.merge(self.df,addBigrams(df.copy(),pairs),how='inner',on='ID')
		self.df = pd.merge(self.df,addTrigrams(df.copy(),trips),how='inner',on='ID')
		#self.df = addTrigrams(df.copy(),trips)

		#self.df = addBigrams(self.df,pairs)
		self.encoder = OneHotEncoder()
		usecols = [c for c in self.df if c != 'ID']
		self.encoder.fit(np.array(self.df.ix[:,usecols],dtype='float'))

	def transformTrain(self,filename):
		df = pd.read_csv(filename,usecols=range(9))
		df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]), zip(['train']*32769, range(32769)))
	
		x = pd.merge(df.ix[:,['ID','ACTION']],self.df,how='left',on='ID',sort=False)
		ignore = ['ID','ACTION']
		usecols = [c for c in x.columns if c not in ignore]
		return self.encoder.transform(np.array(x.ix[:,usecols],dtype='float')), np.array(x.ACTION)

	def transformTest(self,filename):
		df = pd.read_csv(filename)
		df['ID'] = map(lambda x: "%s.%06i"%(x[0],x[1]), zip(['test']*58921, range(58921)))

		x = pd.merge(df.ix[:,['ID','ROLL_CODE']],self.df,how='left',on='ID',sort=False)
		ignore = ['ID','ROLL_CODE']
		usecols = [c for c in x.columns if c not in ignore]
		return self.encoder.transform(np.array(x.ix[:,usecols],dtype='float'))




