import matplotlib
import matplotlib.pyplot as plt
from KNN import autoNorm,file2matrix
from numpy import *

from pylab import mpl

mpl.rcParams['font.sans-serif']=['FangoSong']
mpl.rcParams['axes.unicode_minus']=False

#得到数据
datingDataMat,datingLabels=file2matrix('datingTestSet.txt')

#使用归一
normMat,ranges,minVals=autoNorm(datingDataMat)

#输出数据
#print(datingDataMat)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(normMat[:,1],normMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))

plt.show()
