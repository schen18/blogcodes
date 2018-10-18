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
nrange=[1,10]
threshold=0.4
"""
nrange = lowest/highest random integer

"""
xval=[]
yval=[]

for nsize in range(10,100,5):
    for i in range(100):
        pcorr=[]
        for s in range(100):
            x=np.random.randint(nrange[0],nrange[1],size=nsize)
            y=np.random.randint(nrange[0],nrange[1],size=nsize)
            pcorr.append(np.corrcoef(x,y)[0,1])

        xval.append(nsize)
        pca=np.asarray(pcorr)
        yval.append((pca >= threshold).sum())
        pcorr.clear

# Plot results 
plt.figure(figsize=(8,4))
plt.plot(xval, yval, 'o', color='b', alpha=0.2)
plt.title('Incidence meeting Threshold of p='+str(threshold))
plt.xlabel('Size of Random Variable')

plt.show()