import kMeans
import matplotlib.pyplot as plt
from  numpy import  *
'''
#mat将数组转化为矩阵,之后就可以进行线性代数的一些操作
loadData=kMeans.loadDataSet('testSet.txt')
dataMat=mat(loadData)
dataMat2=array(dataMat)

#调用means函数
myCentroids,clustAssing=kMeans.kMeans(dataMat,4)

dataMat3=array(clustAssing)
plt.figure()
plt.scatter(dataMat2[:,0],dataMat2[:,1])
plt.show()
'''
datMat3=mat(kMeans.loadDataSet('testSet2.txt'))
centList,myNewAssments=kMeans.biKmeans(datMat3,3)
plt.plot(centList[:, 0], centList[:, 1], 'ro')
plt.plot(datMat3[:, 0], datMat3[:, 1], 'bo')
plt.show()
# print(centList)
# print(myNewAssments)