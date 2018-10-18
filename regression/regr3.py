#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
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

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','cylinders','displacement','horsepower','weight','acceleration'])

yvar='mpg'
xvar=['cylinders','displacement','horsepower','weight','acceleration']

X = df[xvar]
Y = df[yvar]
regr = linear_model.LinearRegression()
regr.fit(X, Y)

xval=[]
rmse=[]

for b in range(nsim):
    for s in range(5,nrange):
        idf=df.sample(n=s, replace=True)
        xval.append(nrange-s)
        tdf=pd.DataFrame()
        for col in idf:
            r=np.random.randint(df[col].min(),df[col].max(),nrange-s)
            tdf[col]=pd.Series(r)
    
        ndf=pd.concat([idf,tdf])
        dX = ndf[xvar]
        dY = ndf[yvar]
        y_pred = regr.predict(dX)
        rmse.append(np.sqrt(mean_squared_error(dY, y_pred)))
    
# Plot results as histograms
plt.figure(figsize=(10,6))
#plt.scatter(xval, rmse, c='r', alpha=0.3)
#plt.title('Bootstrap='+str(nsim)+' samples')
#plt.ylabel('RMSE')
#plt.xlabel('Random observations in Test Set n='+str(nrange))

h=sns.jointplot(x=xval, y=rmse, color='m', alpha=0.5, kind="kde")
h.set_axis_labels('x', 'y', fontsize=12)
h.ax_joint.set_ylabel('RMSE')
h.ax_joint.set_xlabel('Test Set of n=5 to '+str(nrange)+' random observations')

plt.show()