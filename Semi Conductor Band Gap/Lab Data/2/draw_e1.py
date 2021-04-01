from scipy import special, optimize
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
    return np.rec.array(data, dtype=dtype)

# now read in your file -- the line below gives examples of datatypes
#mydescr = np.dtype([('column1', 'int32'), ('column2Name', 'uint32'), ('col3', 'uint64'), ('c4', 'float32')])
mydescr = np.dtype([('chan', 'int32'), ('counts', 'int32')])
myrecarray = read_array('calibration_data.csv', mydescr)


# here we actually plot the data
pl.plot(myrecarray.chan,myrecarray.counts)



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

pl.xlabel('Channel', fontsize=16, fontweight='bold')
pl.ylabel('Counts', fontsize=16, fontweight='bold')

# save the plot to a file
pl.savefig('calib.png', bbox_inches='tight')
# display the plot so you can see it
pl.show()









