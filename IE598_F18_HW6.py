#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:57:07 2018

@author: yingzhaocheng
"""

#Using the Iris dataset, with 90% for training and 10% for test and the 
#decision tree model
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
X, y = X_iris, y_iris

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score



rdstate=[1,2,3,4,5,6,7,8,9,10]
in_sample = []
out_sample = []

for k in rdstate:
    #train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,
                                                    random_state=k)
    #using preprocessing to process
    scaler = preprocessing.StandardScaler().fit(X_train)
    DecisionTclassifier = DecisionTreeClassifier(max_depth = 6, criterion = 'gini', random_state = 1)
    DecisionTclassifier.fit(X_train, y_train)
    y_pred_out = DecisionTclassifier.predict(X_test)
    y_pred_in = DecisionTclassifier.predict(X_train)
    out_sample_score = accuracy_score(y_test, y_pred_out)
    in_sample_score = accuracy_score(y_train, y_pred_in)
    in_sample.append(in_sample_score)
    out_sample.append(out_sample_score)
    print('Random State: %d, in_sample: %.3f, out_sample: %.3f'%(k, in_sample_score,
                                                             out_sample_score))




#mean
mean_in = np.mean(in_sample)
#standard devistion
mean_out = np.mean(out_sample)

std_in = np.std(in_sample)
std_out = np.std(out_sample)
print('In sample Mean: %.3f, Out sample Mean: %.3f \nIn sample STD: %.3f,Out sample STD: %.3f'%(mean_in, mean_out, std_in, std_out))

#crossvalidation

cv_scores = cross_val_score(DecisionTclassifier, X_train, y_train, cv = 10)
print('CV Scores:', cv_scores)
print("mean of cv score: {:.3f}".format(np.mean(cv_scores)))
print("variance of cv score: {:.3f}".format(np.var(cv_scores)))
print("std of cv score:{:.5f}".format(np.std(cv_scores)))
y_pred = DecisionTclassifier.predict(X_test)
cvout_sample_score= accuracy_score(y_test, y_pred) 
print("out sample CV accuracy: {:.3f}".format(cvout_sample_score))

print("My name is Zhaocheng Ying")
print("My NetID is: zying4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






