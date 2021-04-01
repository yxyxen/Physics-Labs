# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:56:15 2019

@author: yxyxe
"""
from scipy import optimize
import numpy as np
import pylab as pl
a=[[0,1,222,33],[2,11,2,333],[1,111,22,3]]
b = []
b.append([1,2])
b.append([3,4])
print(b)
c=b[0::]
print(c)
print(np.amin(a, axis = 0))
print(np.argmin(a, axis = 0))
#print(np.append(b, [[0,1,4,5]], axis = 0))
for i in range(4):
    print(i)
b=[1,2,3,4]
print(b*5)
print(9578.7*8.617*10**(-5))
print(np.log(2.72))
print(np.arctan(1))
import numpy as np
from scipy.signal import argrelextrema

x = np.random.random(12)
print(x)
# for local maxima
print(argrelextrema(x, np.greater))

# for local minima
print(argrelextrema(x, np.less))
print(x[argrelextrema(x, np.greater)[0]])