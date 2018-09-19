#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np 

"""
Set parameters below
--------------------

"""
nsim=1000
nsize=50
threshold=0.4
"""
nsim = number of simulations
nsize = number of items in random arrays

"""
xval=[]
yval=[]

for isize in range(5,nsize+1,2):
    for i in range(100):
        pcorr=[]
        for s in range(nsim):
            x=np.random.randint(1,nsize,size=30)
            y=np.random.randint(1,isize+1,size=30)
            pcorr.append(np.corrcoef(x,y)[0,1])

        xval.append(isize)
        pca=np.asarray(pcorr)
        yval.append((pca >= threshold).sum())
        pcorr.clear

# Plot results
plt.plot(xval, yval, 'o', color='g', alpha=0.5)
plt.title('Incidence meeting Threshold of p='+str(threshold))
plt.xlabel('Random Integer range')

plt.show()