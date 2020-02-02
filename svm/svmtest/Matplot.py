import matplotlib.pyplot as plt
import numpy as np
from svmtest import svmMLiA


dataArr,labelArr=svmMLiA.loadDataSet('testSetRBF2.txt')
print(labelArr[:])
dataArr=np.array(dataArr)

plt.figure()
plt.scatter(dataArr[:,0],dataArr[:,1])
plt.show()
