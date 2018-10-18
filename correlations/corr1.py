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
nsize=370
nrange=[0,5]

"""
nsim = number of simulations
nsize = number of items in random arrays
nrange = lowest/highest random integer

"""

pcorr=[]
pscorr=[]

for s in range(nsim):
    x=np.random.randint(nrange[0],nrange[1],size=nsize)
    y=np.random.randint(nrange[0],nrange[1],size=nsize)
    pcorr.append(np.corrcoef(x,y)[0,1])
    # subsets first 30 items from x,y
    xs=x[:30] 
    ys=y[:30] 
    pscorr.append(np.corrcoef(xs,ys)[0,1])

# Plot results as histograms
plt.figure(figsize=(10,5))
plt.title('n='+str(nsim)+' random draws of random integers from '+str(nrange[0])+' to '+str(nrange[1]))
plt.xlabel('Pearson Correlation')

#plt.subplot(2, 1, 1)
#plt.hist(pcorr, bins = 20, color='g')
#plt.ylabel('n='+str(nsize))
#
#plt.subplot(2, 1, 2)
#plt.hist(pscorr, bins = 20)
#plt.ylabel('n=30')

sns.distplot(np.array(pcorr), label='n='+str(nsize))
sns.distplot(np.array(pscorr), label='n=30')

plt.legend()
plt.show()