from scipy import optimize
import numpy as np
import pylab as pl

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
    for i in range(0,3):
        data[i] = data[i][0:]
#    print("data array is ")
#    print(data)
#    print("data array ends here")
    return np.rec.array(data, dtype=dtype)

# now read in your file -- the line below gives examples of datatypes
#mydescr = np.dtype([('column1', 'int32'), ('column2Name', 'uint32'), ('col3', 'uint64'), ('c4', 'float32')])
mydescr = np.dtype([('xpos', 'float32'), ('xerr', 'float32'), ('ypos', 'float32'),('yerr', 'float32')])
myrecarray = read_array('Si_run.csv', mydescr)

# put in a small error on the x measurement
#xerr = 0.00003

# here we plot the data with error bars
pl.errorbar(myrecarray.xpos,myrecarray.ypos,myrecarray.yerr,myrecarray.xerr, fmt = 'o')


# now we want to do a least squares fit to the data -- a straight line
# here is our function that we fit 
def func(x, a, b):
    return a + b*x


# Initial guess for a and b, the parameters of the fit
x0    = np.array([-1.0, 1])
sigma = myrecarray.yerr

result = optimize.curve_fit(func, myrecarray.xpos, myrecarray.ypos, x0, sigma)
print (result)

#plot the least square fitted line.
a = result[0][0]
b = result[0][1]
x1 = np.amin(myrecarray.xpos)
x2 = np.amax(myrecarray.xpos)
pl.plot([x1, x2], [a+b*x1, a+b*x2])

# Change size and font of tick labels
# Again, this doesn't work in interactive mode.
fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

pl.xlabel('1/T', fontsize=16, fontweight='bold')
pl.ylabel('ln(I$_0$)', fontsize=16, fontweight='bold')

# save the plot to a file
#pl.savefig('HEP.png', bbox_inches='tight')
# display the plot so you can see it
pl.show()









