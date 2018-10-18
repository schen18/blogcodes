#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

"""
Set parameters below
--------------------

"""
nsim=100
nrange=50
trees=10

"""
nsim = number of simulations
nrange = size of test set
trees = number of decision trees 
"""

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','displacement','weight','acceleration'])

yvar='mpg'
xvar=['displacement','weight','acceleration']


Y = np.array(df[yvar])
X = np.array(df.drop(yvar,axis=1))

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 50)
rf = RandomForestRegressor(n_estimators=trees, max_depth=len(xvar))
rf.fit(train_features, train_labels);

vdf = pd.DataFrame(np.column_stack((test_labels, test_features)), columns=['mpg','displacement','weight','acceleration'])

xval=[]
accu=[]

for b in range(nsim):
    for s in range(5,nrange):
        idf=vdf.sample(n=s, replace=True)
        xval.append(nrange-s)
        tdf=pd.DataFrame()
        for col in idf:
            r=np.random.randint(df[col].min(),df[col].max(),nrange-s)
            tdf[col]=pd.Series(r)
    
        ndf=pd.concat([idf,tdf])
        
        dY = np.array(ndf[yvar])
        dX = np.array(ndf.drop(yvar,axis=1))
    
        # Use the forest's predict method on the test data
        predictions = rf.predict(dX)
        errors = abs(predictions - dY)
        accuracy = 100 - np.mean(100 * (errors / dY))
        accu.append(round(accuracy, 2))
    
## Plot results as histograms
plt.figure(figsize=(10,6))
##plt.scatter(xval, accu, c='r', alpha=0.3)
##plt.title('Bootstrap='+str(nsim)+' samples')
##plt.ylabel('RMSE')
##plt.xlabel('Random observations in Test Set n='+str(nrange))
#
h=sns.jointplot(x=xval, y=accu, color='c', alpha=0.5, kind="kde")
h.set_axis_labels('x', 'y', fontsize=12)
h.ax_joint.set_ylabel('Accuracy')
h.ax_joint.set_xlabel('Test Set of n=5 to '+str(nrange)+' random observations')

plt.show()