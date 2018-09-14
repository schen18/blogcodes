#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np 

"""
Set parameters below
--------------------

"""
nsim=1000
nhigh=10
threshold=0.4
"""
nsim = number of simulations
nhigh = highest integer in random arrays

"""
xval=[]
yval=[]

for nsize in range(10,101,5):
    for i in range(100):
        pcorr=[]
        for s in range(nsim):
            x=np.random.randint(1,nhigh+1,size=nsize)
            y=np.random.randint(1,nhigh+1,size=nsize)
            pcorr.append(np.corrcoef(x,y)[0,1])

        xval.append(nsize)
        pca=np.asarray(pcorr)
        yval.append((pca >= threshold).sum())
        pcorr.clear

# Plot results 
plt.plot(xval, yval, 'o', color='b', alpha=0.5)
plt.title('Incidence meeting Threshold of p='+str(threshold))
plt.xlabel('Size of Random Variable')

plt.show()