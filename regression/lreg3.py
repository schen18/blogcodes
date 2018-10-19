#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn import linear_model
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
nrange=50
"""
nsim = number of simulations
nrange = size of test set 
"""

csvfile='wdbc.csv'
df=pd.read_csv(csvfile, usecols=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'])

yvar='diagnosis'
xvar=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']

Y = np.array(df[yvar])
X = np.array(df.drop(yvar,axis=1))

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 50)
regr = linear_model.LogisticRegression()
regr.fit(train_features, train_labels);
regr.fit(X, Y)

vdf = pd.DataFrame(np.column_stack((test_labels, test_features)), columns=['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'])

xval=[]
roc=[]
accu=[]

for b in range(nsim):
    for s in range(5,nrange):
        idf=vdf.copy()
        xval.append(nrange-s)
        tdf=idf.sample(n=s)
        tdf.replace({0: 1, 1: 0})
        ndf=pd.concat([idf,tdf])
        
        dY = np.array(ndf[yvar])
        dX = np.array(ndf.drop(yvar,axis=1))
    
        # Use the predict method on the test data
        predictions = regr.predict(dX)
        accu.append(metrics.accuracy_score(dY,predictions))
        roc.append(metrics.roc_auc_score(dY,predictions))

## Plot results as histograms
plt.figure(figsize=(10,6))
##plt.scatter(xval, accu, c='r', alpha=0.3)
##plt.title('Bootstrap='+str(nsim)+' samples')
##plt.ylabel('RMSE')
##plt.xlabel('Random observations in Test Set n='+str(nrange))
#
h=sns.jointplot(x=xval, y=accu, color='peru', alpha=0.5, kind="kde")
h.set_axis_labels('x', 'y', fontsize=12)
h.ax_joint.set_ylabel('Accuracy')
h.ax_joint.set_xlabel('Classification Set with n=5 to '+str(nrange)+' mislabeled observations')

