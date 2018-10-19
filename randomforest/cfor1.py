#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

"""
Set parameters below
--------------------

"""
nsim=100
nrange=[0,10]
nrand=10
trees=100
"""
nsim = number of simulations
nrange = lowest/highest random integer
nrand = number of random variables
trees = number of decision trees 
"""

csvfile='wdbc.csv'
df=pd.read_csv(csvfile, usecols=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'])

yvar='diagnosis'
xvar=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']

roc=[]
accu=[]

if nrand>0:
    for s in range(nsim):
        idf=df.copy()
        for i in range(nrand):
            col_name = "rand" + str(i)
            if col_name not in xvar:
                xvar.append(col_name)
                
            r=np.random.randint(nrange[0],(nrange[1])**(i+1),size=len(df.index))
            idf[col_name] = pd.Series(r)

        X = np.array(idf[yvar])
        Y = np.array(idf.drop(yvar,axis=1))
        
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(Y, X, test_size = 0.25)
        rf = RandomForestClassifier(n_estimators=trees, max_depth=len(xvar))
        rf.fit(train_features, train_labels);
        
        predictions = rf.predict(test_features)
        accu.append(metrics.accuracy_score(test_labels,predictions))
        roc.append(metrics.roc_auc_score(test_labels,predictions))
    
    # Plot results as histograms    
    plt.figure(figsize=(10,6))
#    plt.subplot(2, 1, 1)
#    plt.hist(accu, bins = 20, color='g')
#    plt.ylabel('Accuracy')
#    
#    plt.subplot(2, 1, 2)
#    plt.hist(roc, bins = 20)
#    plt.ylabel('ROC AUC')
#    plt.xlabel('Histograms')
#
#    plt.suptitle("Random Forest Classifier with "+str(nrand)+" random variable/s")
    
    h=sns.jointplot(x=accu, y=roc, color='b', alpha=0.5, kind="hex")
    h.set_axis_labels('x', 'y', fontsize=12)
    h.ax_joint.set_ylabel('Accuracy vs. ROC AUC')
    h.ax_joint.set_xlabel("Random Forest Classifier with "+str(nrand)+" random variable/s")
    
    plt.show()

else:
    X = np.array(df[yvar])
    Y = np.array(df.drop(yvar,axis=1))
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(Y, X, test_size = 0.25)
    
    # Instantiate model with 100 decision trees
    rf = RandomForestClassifier(n_estimators=trees, max_depth=len(xvar))
    rf.fit(train_features, train_labels);
    
    predictions = rf.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(test_labels,predictions))
    print('ROC AUC:', metrics.roc_auc_score(test_labels,predictions))


    
    


