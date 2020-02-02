import adaboost
import matplotlib.pyplot as plt
dataMat,classLabels=adaboost.loadSimpData()

xcord0=[]
ycord0=[]
xcord1=[]
ycord1=[]

for i in range(len(classLabels)):
    if classLabels[i]==-1:
        xcord0.append(dataMat[i,0])
        ycord0.append(dataMat[i,1])
    elif classLabels[i]==1:
        xcord1.append(dataMat[i,0])
        ycord1.append(dataMat[i,1])

plt.figure()
plt.xlim(0.8,2.2)
plt.ylim(0.8,2.2)
plt.scatter(xcord0,ycord0,marker='s')
plt.scatter(xcord1,ycord1,marker='o')

plt.show()
