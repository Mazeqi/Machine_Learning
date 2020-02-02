import adaboost
import  numpy as np

#datMat,classLabels=adaboost.loadSimpData()
#classifierArr=adaboost.adaBoostTrainDS(datMat,classLabels,30)
#print(adaboost.adaClassify([[5,5],[0,0]],classifierArr))
"""
datArr,labelArr=adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray=adaboost.adaBoostTrainDS(datArr,labelArr,10)
print(classifierArray)

testArr,testLabelArr=adaboost.loadDataSet('horseColicTest2.txt')
prediction10=adaboost.adaClassify(testArr,classifierArray)
errArr=np.mat(np.ones((67,1)))

out=errArr[prediction10!=np.mat(testLabelArr).T].sum()
print(out)
"""

datArr,labelArr=adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst=adaboost.adaBoostTrainDS(datArr,labelArr,100)
adaboost.plotROC(aggClassEst.T,labelArr)