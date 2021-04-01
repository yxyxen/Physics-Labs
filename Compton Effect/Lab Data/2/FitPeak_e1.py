from scipy import special, optimize
import numpy as np
import pylab as pl
from lmfit import Model


filename = '165.csv'

pl.rc('axes', linewidth=2)

#function to read in files
def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """

    cast = np.cast
    data = [[] for dummy in range(len(dtype))]
#    j=0
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
#        j += 1
    for i in range(len(dtype)):
        print("test ",i,"  ",dtype[i]," ",data[i])
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)

# now read in the file
mydescr = np.dtype([('binnum', 'int32'), ('count', 'int32')])
myrecarray = read_array(filename, mydescr)

def gaussian(x, amp, cen, wid, const):
    return ((const + amp * np.exp(-(x-cen)**2 / (2*wid**2))))

# set up your gaussian model with initial guess at parameters
mod = Model(gaussian)
pars = mod.make_params(const = 100, amp = 3000, cen = 825, wid = 15)

# assign your data to x and y
x = myrecarray.binnum
y = myrecarray.count


result = mod.fit(y, pars, x=x )
#result = mod.fit(y, x, amp, cen, wid, const )
#result = mod.fit(myrecarray.count, pars, x)
print(result.fit_report())

pl.scatter(myrecarray.binnum, myrecarray.count)

pl.plot(myrecarray.binnum, result.best_fit, color='r')


fontsize = 14
ax = pl.gca()

for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

print("1")
pl.xlim()
pl.xlabel('Channel Number', fontsize=16, fontweight='bold')
pl.ylabel('Counts',fontsize=16, fontweight='bold')
print("2")
pl.savefig('gaussfit.png', bbox_inches='tight')
print("3")
pl.show()
print("4")
