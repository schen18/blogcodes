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
nsim=1000
prand=0.01
trees=10
"""
nsim = number of simulations
prand = percentage of random observations
trees = number of decision trees 
"""

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','cylinders','displacement','horsepower','weight','acceleration'])

yvar='mpg'
xvar=['cylinders','displacement','horsepower','weight','acceleration']

mae=[]
accu=[]
    
for s in range(nsim):
    idf=df.copy()
    tdf=pd.DataFrame()
    for col in idf:
        r=np.random.randint(idf[col].min(),idf[col].max(),int(len(idf.index)*prand))
        tdf[col]=pd.Series(r)

    ndf=pd.concat([idf,tdf])
    
    X = np.array(ndf[yvar])
    Y = np.array(ndf.drop(yvar,axis=1))
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(Y, X, test_size = 0.25)
    rf = RandomForestRegressor(n_estimators=trees, max_depth=len(xvar))
    rf.fit(train_features, train_labels);
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    accuracy = 100 - np.mean(100 * (errors / test_labels))
    mae.append(round(np.mean(errors), 2))
    accu.append(round(accuracy, 2))
    
# Plot results as histograms
plt.figure(figsize=(8,6))
plt.subplot(2, 1, 1)
plt.hist(mae, bins = 20, color='g')
plt.title('Random Forest Bootstrap='+str(nsim)+' '+str(prand)+'% Rnd')
plt.ylabel('MAE')

plt.subplot(2, 1, 2)
plt.hist(accu, bins = 20)
plt.ylabel('Accuracy')
plt.xlabel('Histograms')

plt.show()