import scipy
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from scipy.signal import argrelextrema
pl.rc('axes', linewidth=2)

# set up your read_array to use later to read in your file
def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = np.cast
    data = [[] for dummy in range(len(dtype))]
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in range(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)

# now read in your file -- the line below gives examples of datatypes
#mydescr = np.dtype([('column1', 'int32'), ('column2Name', 'uint32'), ('col3', 'uint64'), ('c4', 'float32')])
mydescr = np.dtype([('xpos', 'float32'), ('xerr', 'float32'), ('ypos', 'float32'),('yerr', 'float32')])
def column(matrix, i):
    #extracting column i in a matrix.
    return [row[i] for row in matrix]
#draw the single slit and double slits results
fileList = ['Left.csv','Right.csv','Double.csv','Light_bulb.csv']
maxPos = [2.05, 3.85, 2.5, 2.55]
maxInt = [16.7, 6.4, 52.3, 980.25]
arrayL = [0.00067,0.00067,0.00067,0.000546]
#
myrecarray = read_array('Left.csv', mydescr)
pl.errorbar(myrecarray.xpos,myrecarray.ypos,myrecarray.yerr,myrecarray.xerr, fmt = 'o',label = 'experimental data')
y0 = 2.05
I0 = 16.7
l = 0.00067
def func(y):
    theta = np.arctan((y-y0)/500)
    sigma = 2*np.pi*0.085*np.sin(theta)/l
    return I0*((np.sin(sigma/2))**2)/((sigma/2)**2)
y = np.arange(0,7,0.001)
pl.plot(y,func(y),label = 'theoretical function')

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Voltage(mV)$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper right')
pl.show()

#
myrecarray = read_array('Right.csv', mydescr)    
pl.errorbar(myrecarray.xpos,myrecarray.ypos,myrecarray.yerr,myrecarray.xerr, fmt = 'o',label = 'experimental data')

y0 = 3.85
I0 = 6.4
l = 0.00067
def func(y):
    theta = np.arctan((y-y0)/500)
    sigma = 2*np.pi*0.085*np.sin(theta)/l
    return I0*((np.sin(sigma/2))**2)/((sigma/2)**2)
y = np.arange(0,7,0.001)
pl.plot(y,func(y),label = 'theoretical function')

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Voltage(mV)$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper right')
pl.show()

#
myrecarray = read_array('Double.csv', mydescr)    
pl.errorbar(myrecarray.xpos,myrecarray.ypos,myrecarray.yerr,myrecarray.xerr, fmt = 'o',label = 'experimental data')

y0 = 2.5
I0 = 52.3
l = 0.00067
def func(y):
    theta = np.arctan((y-y0)/500)
    alpha = (np.pi*0.085*np.sin(theta))/l
    beta = (np.pi*0.353*np.sin(theta))/l
    return I0*((np.cos(beta))**2)*((np.sin(alpha)/alpha)**2)
y = np.arange(0,6,0.001)
pl.plot(y,func(y),label = 'theoretical function')

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Voltage(mV)$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper right')
pl.show()
#

exI = scipy.signal.find_peaks(myrecarray.ypos,height = 5, distance = 4)
exI = np.delete(exI[0],2,0)
localEx = [myrecarray[el] for el in exI]
xErr = [xE*10 for xE in column(localEx,1)]
pl.errorbar(column(localEx,0),column(localEx,2),column(localEx,3),xErr, fmt = ',',label = 'experimental maxima')
print("experiment maxima:", column(localEx,0))
exI = scipy.signal.find_peaks([-el for el in myrecarray.ypos],height = -5, distance = 4)
localEx = [myrecarray[el] for el in exI[0]]
xErr = [xE*10 for xE in column(localEx,1)]
pl.errorbar(column(localEx,0),column(localEx,2),column(localEx,3),xErr, fmt = ',',label = 'experimental minima')
print("experiment minima:",column(localEx,0))

exI = scipy.signal.find_peaks([func(el) for el in y],height = 5, distance = 4)
thLocalEx = [y[el] for el in exI[0]]
pl.plot(thLocalEx,[func(el) for el in thLocalEx],'.',label = 'theoretical maxima')
print("theoretical maxima:",thLocalEx)
exI = scipy.signal.find_peaks([-func(el) for el in y],height = -5, distance = 4)
thLocalEx = [y[el] for el in exI[0]]
pl.plot(thLocalEx,[func(el) for el in thLocalEx],'.',label = 'theoretical minima')
print("theoretical minima:",thLocalEx)

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Voltage(mV)$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
pl.show()

#
myrecarray = read_array('Light_bulb.csv', mydescr)    
pl.errorbar(myrecarray.xpos,myrecarray.ypos,myrecarray.yerr,myrecarray.xerr, fmt = 'o',label = 'experimental data')

y0 = 2.55
I0 = 980.25
Id = 7.5 #dark rate
l = 0.000546
def func(y):
    theta = np.arctan((y-y0)/500)
    alpha = (np.pi*0.085*np.sin(theta))/l
    beta = (np.pi*0.353*np.sin(theta))/l
    return (I0-Id)*((np.cos(beta))**2)*((np.sin(alpha)/alpha)**2) + Id
y = np.arange(0,5.5,0.001)
pl.plot(y,func(y),label = 'theoretical function')

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Count/s$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,loc='upper right')
pl.show()
#    

exI = scipy.signal.find_peaks(myrecarray.ypos,height = 100, distance = 4)
exI = np.delete(exI[0],3,0)
localEx = [myrecarray[el] for el in exI]
xErr = [xE*10 for xE in column(localEx,1)]
pl.errorbar(column(localEx,0),column(localEx,2),column(localEx,3),xErr, fmt = ',',label = 'experimental maxima')
print("experiment maxima:", column(localEx,0))
exI = scipy.signal.find_peaks([-el for el in myrecarray.ypos],height = -100, distance = 4)
localEx = [myrecarray[el] for el in exI[0]]
xErr = [xE*10 for xE in column(localEx,1)]
pl.errorbar(column(localEx,0),column(localEx,2),column(localEx,3),xErr, fmt = ',', label = 'experimental minima')
#plt.legend(handles = exMin)
print("experiment minima:",column(localEx,0))

exI = scipy.signal.find_peaks([func(el) for el in y],height = 100, distance = 4)
exI = np.delete(exI[0],3,0)
thLocalEx = [y[el] for el in exI]
pl.plot(thLocalEx,[func(el) for el in thLocalEx],'.',label = 'theoretical maxima')
print("theoretical maxima:",thLocalEx)
exI = scipy.signal.find_peaks([-func(el) for el in y],height = -50, distance = 4)
thLocalEx = [y[el] for el in exI[0]]
pl.plot(thLocalEx,[func(el) for el in thLocalEx],'.',label = 'theoretical minima')
#plt.legend(handles=thMin)
print("theoretical minima:",thLocalEx)

fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$Position(mm)$', fontsize=16, fontweight='bold')
pl.ylabel('$Count/s$', fontsize=16, fontweight='bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
pl.show()