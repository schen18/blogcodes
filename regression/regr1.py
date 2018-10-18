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
nsim=1000
nrange=[0,10]
nrand=1

"""
nsim = number of simulations
nrange = lowest/highest random integer
nrand = number of random variables

"""

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','cylinders','displacement','horsepower','weight','acceleration'])

yvar='mpg'
xvar=['cylinders','displacement','horsepower','weight','acceleration']

pcorr=[]
rsq=[]
rcoef=[]
rmse=[]

if nrand>0:
    for s in range(nsim):
        idf=df.copy()
        for i in range(nrand):
            if len(pcorr)<=i:
                pcorr.append([])
                rcoef.append([])
                
            col_name = "rand" + str(i)
            if col_name not in xvar:
                xvar.append(col_name)
                
            r=np.random.randint(nrange[0],(nrange[1])**(i+1),size=len(df.index))
            idf[col_name] = pd.Series(r)
            
            pcorr[i].append(idf[yvar].corr(idf[col_name]))

        X = idf[xvar]
        Y = idf[yvar]
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
            
        y_pred = regr.predict(X)
        rsq.append(r2_score(Y, y_pred))
        rmse.append(np.sqrt(mean_squared_error(Y, y_pred)))
        for i in range(nrand):
            rcoef[i].append(regr.coef_[len(regr.coef_)-1*(i+1)])
            
    # Plot results as histograms
    plt.figure(figsize=(10,6))
    plt.subplot(2, 2, 1)
    plt.hist(rsq, bins = 20, color='g')
    plt.title('Bootstrap='+str(nsim)+', Rnd='+str(nrand))
    plt.ylabel('R squared')
    
    plt.subplot(2, 2, 2)
    plt.title('Scatter r['+str(nrand)+'] vs. R2')
    colors = np.random.rand(len(rsq))
    plt.scatter(pcorr[nrand-1], rsq, c=colors, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(rmse, bins = 20)
    plt.ylabel('RMSE')
    plt.xlabel('Histograms')
    
    plt.subplot(2, 2, 4)
    colors = np.random.rand(len(rmse))
    plt.scatter(rsq, rmse, c=colors, alpha=0.3)
    plt.xlabel('Scatter R vs. RMSE')

    plt.suptitle("Linear Regression")
    plt.show()

else:
    X = df[xvar]
    Y = df[yvar]
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    
    y_pred = regr.predict(X)
    
    print(r2_score(Y, y_pred))
    print(regr.coef_)

