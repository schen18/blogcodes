#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
sns.set()

"""
Set parameters below
--------------------

"""
nsim=1000
nsize=30
threshold=0.4
"""
nsim = number of simulations
nsize = number of items in random arrays

"""
xval=[]
yval=[]

for isize in range(5,50,2):
    for i in range(int(nsim/10)):
        pcorr=[]
        for s in range(nsim):
            x=np.random.randint(1,isize,size=nsize)
            y=np.random.randint(1,isize,size=nsize)
            pcorr.append(np.corrcoef(x,y)[0,1])
  
        xval.append(isize)        
        pca=np.asarray(pcorr)
        yval.append((pca >= threshold).sum())
        pcorr.clear

# Plot results
plt.figure(figsize=(8,4))
plt.plot(xval, yval, 'o', color='r', alpha=0.2)
plt.title('Incidence meeting Threshold of p='+str(threshold))
plt.xlabel('Bootstrap Range of Random Integers of n='+str(nsize))

plt.show()