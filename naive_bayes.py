#!/usr/bin/env python

__author__ = 'Miroslaw Horbal'
__email__ = 'miroslaw@gmail.com'
__date__ = '12-06-2013'

import numpy as np
import pandas as pd 
import pylab as pl
from sklearn.cross_validation import KFold
import plotting
reload(plotting)
from sklearn.metrics import roc_curve, auc

# Utility functions 
def doubs_generator(n):
    """
    Generate doubles (i,j) with i<j from 0 to n
    """
    for i in range(n):
        for j in range(i+1,n):
            yield (i,j)
            
def trips_generator(n):
    """
    Generate triples (i,j,k) with i<j<k from 0 to n
    """
    for i,j in doubs_generator(n):
        for k in range(j+1,n):
            yield (i,j,k)
                
def group_data(data, hash=hash, generator=trips_generator):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in generator(n):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def extract_counts(L):
    """
    Take a 1D numpy array as input and return a dict mapping values to counts
    """
    uniques = set(list(L))
    counts = dict((u, np.sum(L==u)) for u in uniques)
    return counts 
    
class NaiveBayesClassifier(object):
    """
    Naive Bayes Classifier with additive smoothing
    
    Params
        :alpha - hyperparameter for additive smoothing
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __repr__(self):
        return 'NaiveBayesClassifier(alpha=%.1e)' % (self.alpha)
    
    def fit(self, X, y):
        """
        Trains Naive Bayes Classifier on data X with labels y
        
        Input
            :X - numpy.array with shape (num_points, num_features)
            :y - numpy.array with shape (num_points, )
        
        Sets attributes
            :pos_prior - estimate of prior probability of label 1
            :neg_prior - estimate of prior probability of label 0
        """
        self._pos_counts = [extract_counts(L) for L in X[y==1].T]
        self._neg_counts = [extract_counts(L) for L in X[y==0].T]
        self._total_pos = float(sum(y==1))
        self._total_neg = float(sum(y==0))
        total = self._total_pos + self._total_neg
        self.pos_prior = self._total_pos / total
        self.neg_prior = self._total_neg / total
        
    def log_predict(self, X):
        """
        Returns log ((P(c=1) / P(c=0)) * prod_i P(x_i | c=1) / P(x_i | c=0))
        using additive smoothing
        
        Input
            :X - numpy.array with shape (num_points, num_features)
                 num_features must be the same as data used to fit model
        """
        m,n = X.shape
        if n != len(self._pos_counts):
            raise Error('Dimension mismatch: expected %i features, got %i' % (
                         len(self._pos_counts), n))
        alpha = self.alpha
        tot_neg = self._total_neg
        tot_pos = self._total_pos
        preds = np.zeros(m)
        for i, xi in enumerate(X):
            Pxi_neg = np.zeros(n)
            Pxi_pos = np.zeros(n)
            for j,v in enumerate(xi):
                nc = self._neg_counts[j].get(v,0)
                pc = self._pos_counts[j].get(v,0)
                nneg = len(self._neg_counts[j])
                npos = len(self._pos_counts[j])
                # Compute probabilities with additive smoothing
                Pxi_neg[j] = (nc + alpha) / (tot_neg + alpha * nneg)
                Pxi_pos[j] = (pc + alpha) / (tot_pos + alpha * npos)
            # Compute log pos / neg class ratio
            preds[i] = np.log(self.pos_prior) + np.sum(np.log(Pxi_pos)) - \
                       np.log(self.neg_prior) - np.sum(np.log(Pxi_neg))
        return preds 

    def predict(self, X, cutoff=0):
        """
        Returns predicted binary classes for data with decision boundry given
        by cutoff 
        
        Input
            :X - see NaiveBayesClassifier.log_predict 
            :cutoff - decision boundry for log predictions 
        """
        preds = self.log_predict(X)
        return (preds >= cutoff).astype(int)
    
def main(train_file='train.csv', test_file='test.csv', output_file='nb_predict.csv'):
    # Load data
    print 'Loading data...'
    #base = [3, 4, 17, 10, 123, 129, 292, 32, 9, 2, 7, 76, 5, 429, 663, 427, 308, 594, 13, 22, 279, 107, 360, 15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
    base = [15, 557, 16, 583, 431, 23, 19, 724, 27, 425]
    #base = [183, 266, 211, 178, 70, 363, 329, 248, 331, 327, 368,365,66,184, 7, 261, 234, 47, 322, 216, 310,177,127,9,38,18,262,313,343,180,35,362,267,174,254,12,182,205,382,0,292]
    allData = pd.read_csv(train_file,usecols=base)
    
    X = np.array(allData.ix[:32769,:])
    X_test = np.array(allData.ix[32769:,:])
    y = np.array(pd.read_csv('../data/train.csv',usecols=[0]).ACTION)
    #X = np.array(train_data.ix[:,1:-1])     # Ignores ACTION, ROLE_CODE
    #X_test = np.array(test_data.ix[:,1:-1]) # Ignores ID, ROLE_CODE
    
    # Convert features to triples
    print 'Transforming data...'
    #X = group_data(X)
    #X_test = group_data(X_test)
    model = NaiveBayesClassifier(alpha=1e-10)
    
    nFolds = 10
    kf = KFold(len(y),n_folds=nFolds,indices=False,shuffle=True,random_state=1337)
    y_ = np.empty(len(y))
    mean_auc = 0.
    for train, test in kf:
        model.fit(X[train,:],y[train])
        y_[test] = model.log_predict(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], y_[test])
        thisAuc = auc(fpr,tpr)
        print "AUC: %f" % thisAuc
        mean_auc += thisAuc

    print mean_auc/len(kf)
    plotting.PlotROC(y,y_,printAuc=True)
    pl.show()
    np.savetxt('nbTrain710.csv',y_,delimiter=',')

    # Train model 
    print 'Training Naive Bayes Classifier...'
    model.fit(X, y)
    
    # Make prediction on test set
    print 'Predicting on test set...'
    preds = model.log_predict(X_test)
    
    print 'Writing predictions to %s...' % (output_file)
    create_test_submission(output_file, preds)

    return model
    
if __name__=='__main__':
    args = { 'train_file':  '../data/quad10Fractions.csv',
             'test_file':   'test.csv',
             'output_file': 'nb_predict710.csv' }
    model = main(**args)

