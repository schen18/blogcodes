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
nrange=[0,10]
nrand=6
trees=10
"""
nsim = number of simulations
nrange = lowest/highest random integer
nrand = number of random variables
trees = number of decision trees 
"""

csvfile='autompg.csv'
df=pd.read_csv(csvfile, usecols=['mpg','cylinders','displacement','horsepower','weight','acceleration'])

yvar='mpg'
xvar=['cylinders','displacement','horsepower','weight','acceleration']

pcorr=[]
mae=[]
accu=[]

if nrand>0:
    for s in range(nsim):
        idf=df.copy()
        for i in range(nrand):
            if len(pcorr)<=i:
                pcorr.append([])
                
            col_name = "rand" + str(i)
            if col_name not in xvar:
                xvar.append(col_name)
                
            r=np.random.randint(nrange[0],(nrange[1])**(i+1),size=len(df.index))
            idf[col_name] = pd.Series(r)
            
            pcorr[i].append(idf[yvar].corr(idf[col_name]))


        Y = np.array(idf[yvar])
        X = np.array(idf.drop(yvar,axis=1))
        
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.25)
        rf = RandomForestRegressor(n_estimators=trees, max_depth=len(xvar))
        rf.fit(train_features, train_labels);
        
        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)
        errors = abs(predictions - test_labels)
        accuracy = 100 - np.mean(100 * (errors / test_labels))
        mae.append(round(np.mean(errors), 2))
        accu.append(round(accuracy, 2))
    
    # Plot results as histograms
    plt.figure(figsize=(10,6))
    plt.subplot(2, 2, 1)
    plt.hist(mae, bins = 20, color='g')
    plt.title('Bootstrap='+str(nsim)+', Rnd='+str(nrand))
    plt.ylabel('MAE')
    
    plt.subplot(2, 2, 2)
    plt.title('Scatter r['+str(nrand)+'] vs. MAE')
    colors = np.random.rand(len(mae))
    plt.scatter(pcorr[nrand-1], mae, c=colors, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(accu, bins = 20)
    plt.ylabel('Accuracy')
    plt.xlabel('Histograms')
    
    plt.subplot(2, 2, 4)
    colors = np.random.rand(len(mae))
    plt.scatter(mae, accu, c=colors, alpha=0.3)
    plt.xlabel('Scatter MAE vs. Accuracy')

    plt.suptitle("Random Forests with "+str(nrand)+" random variable/s")
    plt.show()

else:
    Y = np.array(df[yvar])
    X = np.array(df.drop(yvar,axis=1))
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.25)
    
    # Instantiate model with 100 decision trees
    rf = RandomForestRegressor(n_estimators=100, max_depth=len(xvar))
    rf.fit(train_features, train_labels);
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate ean absolute error (mae)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    # Calculate mean absolute percentage error (MAPE) & Accuracy
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    

