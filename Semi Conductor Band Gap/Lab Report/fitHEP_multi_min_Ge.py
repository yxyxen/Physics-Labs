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
    #extracting column i in a matrix.
    return [row[i] for row in matrix]
def func(x, a, b):
    #for fitting eq(1) and eq(2)
    return a + b*x   
def func2(x,a,b):
    #used for fitting the data to eq(3)
    return a+(1.5*np.log(x))+(b*x)
def func3(x,a,b):
    return a+np.log(np.exp(b*x)-1)
fileList = ['Ge_boiling.csv', 'Ge_room2.csv', 'Ge_ice.csv', 'Ge_dryice.csv']
temperature = [0.002675,0.003394,0.003619,0.004831]

egData = [] #store the found smallest relative interception
for f in fileList:
    orecarray = read_array(f, mydescr)
    dLen = len(orecarray)
    total = [] #store fit result

#select the data range we want to fit
    if f == fileList[0]:
        p1 = 5
        p2 = dLen-10
        p3 = 14
        #t = temperature[0]
    elif f == fileList[1]:
        p1 = 3
        p2 = dLen-5
        p3 = 8
        #t = temperature[1]
    elif f == fileList[2]:
        p1 = 4
        p2 = dLen-10
        p3 = 10
        #t = temperature[2]
    elif f == fileList[3]:
        p1 = 4
        p2 = dLen-10
        p3 = 10
        #t = temperature[3]
    #print (t)
#    def func3(x,a):
#        return a+np.log(np.exp(1.602*(10**(-19))/(1.38*(10**(-23)))*t*x)-1)
#find a fit that has smallest relative error on the y interception    
    for x in range(p1,p2):
        for y in range(x+p3,p2):
            myrecarray = orecarray[x:y]
 
            x0    = np.array([-1.0,1]) # Initial guess for a and b, the parameters of the fit
            sigma = myrecarray.yerr
            result = optimize.curve_fit(func3, myrecarray.xpos, myrecarray.ypos, x0, sigma)
            a = result[0][0]
            da = np.sqrt(result[1][0][0]) #error on interception
            b = result[0][1]
            db = np.sqrt(result[1][1][1]) #error on slope
            c = np.abs(da/a) #relative error on interception
            total.append([c,a,da,b,db,x,y])
            #total.append([c,a,da,x,y])
    r =  total[(np.argmin(total, axis = 0))[0]]    #[c,a,da,b,db,x,y] with smallest c   
    mina = total[(np.argmin(total,axis = 0))[1]][1]
    maxa = total[(np.argmax(total,axis = 0))[1]][1]
    nda = np.sqrt(r[2]**2+((maxa - mina)/2)**2)
    nr = [r[0],r[1],nda,r[3],r[4],r[5],r[6]]
    #nr = [r[0],r[1],nda,r[3],r[4]]
    egData.append(nr)    
    print(f)
    print('min(d(ln(I\N{SUBSCRIPT ZERO}))/ln(I\N{SUBSCRIPT ZERO})):',r)
    print('min(ln(I\N{SUBSCRIPT ZERO})):',mina)
    print('max(ln(I\N{SUBSCRIPT ZERO})):',maxa)
    print('ln(I\N{SUBSCRIPT ZERO}) = ',r[1],'\u00B1',nda)
    pl.figure(1)
    pl.errorbar(orecarray.xpos[r[5]:r[6]],orecarray.ypos[r[5]:r[6]],orecarray.yerr[r[5]:r[6]],orecarray.xerr[r[5]:r[6]], fmt = 'o')
    #pl.errorbar(orecarray.xpos[r[3]:r[4]],orecarray.ypos[r[3]:r[4]],orecarray.yerr[r[3]:r[4]],orecarray.xerr[r[3]:r[4]], fmt = 'o')
    a = r[1]
    b = r[3]
    x1 = (-orecarray.xpos[r[6]-1] + 11*orecarray.xpos[r[5]])*0.1
    x2 = (11*orecarray.xpos[r[6]-1] - orecarray.xpos[r[5]])*0.1
    #x1 = (-orecarray.xpos[r[4]-1] + 11*orecarray.xpos[r[3]])*0.1
    #x2 = (11*orecarray.xpos[r[4]-1] - orecarray.xpos[r[3]])*0.1
    xp = np.arange(x1,x2,np.abs(x2-x1)/1000)
    pl.plot(xp, func3(xp,a,b))
    
#add labels and stuff
fontsize = 14
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
pl.xlabel('$V$', fontsize=16, fontweight='bold')
pl.ylabel('$lnI$', fontsize=16, fontweight='bold')
pl.show()

#fitting the data with eq(2)
x0 = np.array([10,-7891])
sigma = column(egData,2)
result = optimize.curve_fit(func, [0.002675,0.003394,0.003619,0.004831], column(egData,1), x0, sigma)
print('Eg Result with eq(2)',result)
pl.figure(2)
pl.errorbar([0.002675,0.003394,0.003619,0.004831],column(egData,1),sigma,[0.00002675,0.00003394,0.00003619,0.00004831], fmt = 'o')  
a = result[0][0]
b = result[0][1]
#plot the fit line, slope = -Eg/k
x1 = (-np.amax(temperature) + 11*np.amin(temperature))*0.1
x2 = (11*np.amax(temperature) - np.amin(temperature))*0.1
pl.plot([x1, x2], [a+b*x1, a+b*x2])
print('slope = ',result[0][1], '\u00B1',np.sqrt(result[1][1][1])) #print slope
print('Eg =', -1*b*8.617*10**(-5),'\u00B1',np.sqrt(result[1][1][1])*8.617*10**(-5)) #print Eg

fontsize = 12
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

pl.xlabel('$T^{-1}$', fontsize=16, fontweight='bold')
pl.ylabel('$lnI$', fontsize=16, fontweight='bold')
pl.show()

#from here is the code to fit the data with eq(3)
result = optimize.curve_fit(func2, [0.002675,0.003394,0.003619,0.004831], column(egData,1), x0, sigma)
print('Eg Result with eq(3)',result)
pl.figure(2)
pl.errorbar([0.002675,0.003394,0.003619,0.004831],column(egData,1),sigma,[0.00002675,0.00003394,0.00003619,0.00004831], fmt = 'o')  
a = result[0][0]
b = result[0][1]
#plot the fit line, slope = -Eg/k
x1 = (-np.amax(temperature) + 11*np.amin(temperature))*0.1
x2 = (11*np.amax(temperature) - np.amin(temperature))*0.1
xp = np.arange(x1,x2,np.abs(x2-x1)/1000)
pl.plot(xp, func2(xp,a,b),'r--')
print('slope = ',result[0][1], '\u00B1',np.sqrt(result[1][1][1])) #print slope
print('Eg =', -1*b*8.617*10**(-5),'\u00B1',np.sqrt(result[1][1][1])*8.617*10**(-5)) #print Eg

fontsize = 12
ax = pl.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

pl.xlabel('$T^{-1}$', fontsize=16, fontweight='bold')
pl.ylabel('$lnI$', fontsize=16, fontweight='bold')
pl.show()







