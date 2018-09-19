#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np 

"""
Set parameters below
--------------------

"""
nsim=1000
nsize=100
nlow=0
nhigh=4

"""
nsim = number of simulations
nsize = number of items in random arrays
nhigh = highest integer in random arrays

"""

pcorr=[]
pscorr=[]

for s in range(nsim):
    x=np.random.randint(nlow,nhigh+1,size=nsize)
    y=np.random.randint(nlow,nhigh+1,size=nsize)
    pcorr.append(np.corrcoef(x,y)[0,1])
    # extracts first 30 items from x,y and calculates correlation
    xs=x[:30] 
    ys=y[:30] 
    pscorr.append(np.corrcoef(xs,ys)[0,1])

# Plot results as histograms
plt.subplot(2, 1, 1)
plt.hist(pcorr, bins = 20, color='g')
plt.title('Number of random draws='+str(nsim)+' from '+str(nlow)+' to '+str(nhigh))
plt.ylabel('n='+str(nsize))

plt.subplot(2, 1, 2)
plt.hist(pscorr, bins = 20)
plt.xlabel('Pearson Correlation')
plt.ylabel('n=30')

plt.show()