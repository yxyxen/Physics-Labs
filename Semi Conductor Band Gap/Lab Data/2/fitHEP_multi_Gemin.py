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
    return np.rec.array(data, dtype=dtype)

# now read in your file -- the line below gives examples of datatypes
#mydescr = np.dtype([('column1', 'int32'), ('column2Name', 'uint32'), ('col3', 'uint64'), ('c4', 'float32')])
mydescr = np.dtype([('xpos', 'float32'), ('xerr', 'float32'), ('ypos', 'float32'),('yerr', 'float32')])
def column(matrix, i):
    return [row[i] for row in matrix]
def func(x, a, b):
    return a + b*x  
temperature = [0.002675,0.003394,0.003619,0.004831]

fileList = ['Si_boiling.csv', 'Si_room.csv', 'Si_ice.csv', 'Si_dryice.csv']

egData = [] #store the found smallest relative interception
for f in fileList:
    orecarray = read_array(f, mydescr)
    dLen = len(orecarray)
    total = [] #store fit result
    

#select the data range we want to fit
    if f == fileList[0]:
        p1 = 3
        p2 = dLen-5
        p3 = 15
        t = temperature[0]
    elif f == fileList[1]:
        p1 = 3
        p2 = dLen-5
        p3 = 15
        t = temperature[1]
    elif f == fileList[2]:
        p1 = 3
        p2 = dLen-5
        p3 = 15
        t = temperature[2]
    elif f == fileList[3]:
        p1 = 3
        p2 = dLen-5
        p3 = 15
        t = temperature[3]
    def func2(x, a):
        return a+np.log(np.exp(1.602*(10**(-19))/(1.38*(10**(-23)))*t*x)-1)
        
#find a fit that has smallest relative error on the y interception    
    for x in range(p1,p2):
        for y in range(x+p3,p2):
            myrecarray = orecarray[x:y]
 
            x0    = np.array([-1.0]) # Initial guess for a and b, the parameters of the fit
            sigma = myrecarray.yerr
            result = optimize.curve_fit(func2, myrecarray.xpos, myrecarray.ypos, x0, sigma)
            a = result[0][0]
            da = np.sqrt(result[1][0][0]) #error on interception
            #b = result[0][1]
            #db = np.sqrt(result[1][1][1]) #error on slope
            c = np.abs(da/a) #relative error on interception
            total.append([c,a,da,x,y])
    r =  total[(np.argmin(total, axis = 0))[0]]    #[c,a,da,b,db,x,y] with smallest c   
    egData.append(r)    
    print(f,r)
    pl.figure(1)
    pl.errorbar(orecarray.xpos[r[3]:r[4]],orecarray.ypos[r[3]:r[4]],orecarray.yerr[r[3]:r[4]],orecarray.xerr[r[3]:r[4]], fmt = 'o')
pl.show()

x0 = np.array([10,-7891])
sigma = column(egData,2)  
result = optimize.curve_fit(func, [0.002675,0.003394,0.003619,0.004831], column(egData,1), x0, sigma)
print('EgResult',result)
pl.figure(2)
pl.errorbar([0.002675,0.003394,0.003619,0.004831],column(egData,1),sigma,[0.00002675,0.00003394,0.00003619,0.00004831], fmt = 'o')  
a = result[0][0]
b = result[0][1]
#plot the fit line, slope = -Eg/k
x1 = (-np.amax([0.002675,0.003394,0.003619,0.004831]) + 11*np.amin([0.002675,0.003394,0.003619,0.004831]))*0.1
x2 = (11*np.amax([0.002675,0.003394,0.003619,0.004831]) - np.amin([0.002675,0.003394,0.003619,0.004831]))*0.1
pl.plot([x1, x2], [a+b*x1, a+b*x2]) 
print(-1*b*8.617*10**(-5)) #print Eg

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

pl.xlabel('V', fontsize=16, fontweight='bold')
pl.ylabel('lnI', fontsize=16, fontweight='bold')

# save the plot to a file
#pl.savefig('HEP.png', bbox_inches='tight')
# display the plot so you can see it
pl.show()









