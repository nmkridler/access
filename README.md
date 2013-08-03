access
======

This is my code for 15th place in the Amazon Access competition on Kaggle.

features.py uses fileio.py to generate data and then run a classifier
it also does feature selection

the final blend was 4 logistic regressions with C=2.3 and feature selection on 
tripsFractions.csv with seeds: 1337, 410, 622, 918
+ Miroslaw's naive bayes using the 410 features
+ SGD (parameters in submissions.csv)
+ GBM with conditional probabilities + raw data

There's also code for blending based on Caruana's blending paper. Always overfit. 
