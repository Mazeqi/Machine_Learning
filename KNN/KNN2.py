import numpy as np
import operator

def classify0(inX,dataSet,labels,k):
    #得到数据的组数
    dataSize=dataSet.shape[0]

    #得到分类数据组,用于计算向量
    sortedData=np.tile(inX,(dataSize,1))

    #计算向量
    VectorData=sortedData-dataSet

    #计算平方
    sqVectorData=VectorData**2

    #计算行数据的和,得到距离的平方
    SumsqVector=sqVectorData.sum(axis=1)

    #计算距离,即开方
    SqrtSum=SumsqVector**0.5

    print(SqrtSum)

    #排序,这里的argsort返回的是索引值
    SortSqrtSum=SqrtSum.argsort()

    print(SortSqrtSum)

    #选择前k个点
    classCount={}

    for i in range(k):
        #形成新的标签
        newLabel=labels[SortSqrtSum[i]]
        #计算相同代表距离的标签出现的次数
        classCount[newLabel]=classCount.get(newLabel,0)+1

    sortedclassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedclassCount)
    return sortedclassCount[0][0]
