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
prand=0.01

"""
nsim = number of simulations
prand = percentage of random observations

"""

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','cylinders','displacement','horsepower','weight','acceleration'])

yvar='mpg'
xvar=['cylinders','displacement','horsepower','weight','acceleration']

rsq=[]
rcoef=[]
rmse=[]

    
for s in range(nsim):
    idf=df.copy()
    tdf=pd.DataFrame()
    for col in idf:
        r=np.random.randint(idf[col].min(),idf[col].max(),int(len(idf.index)*prand))
        tdf[col]=pd.Series(r)

    ndf=pd.concat([idf,tdf])
    X = ndf[xvar]
    Y = ndf[yvar]
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
            
    y_pred = regr.predict(X)
    rsq.append(r2_score(Y, y_pred))
    rmse.append(np.sqrt(mean_squared_error(Y, y_pred)))
    
# Plot results as histograms
plt.figure(figsize=(10,6))
plt.subplot(2, 1, 1)
plt.hist(rsq, bins = 20, color='g')
plt.title('Bootstrap='+str(nsim)+' '+str(prand)+'% Rnd')
plt.ylabel('R squared')

plt.subplot(2, 1, 2)
plt.hist(rmse, bins = 20)
plt.ylabel('RMSE')
plt.xlabel('Histograms')

plt.show()