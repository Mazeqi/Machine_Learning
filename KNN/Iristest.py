import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import operator

#inx是测试集,dataSet是训练集,labels是训练集对应的标签
def classify0(inX,dataSet,labels,k):

    #得到数据的组数
    dataSize=dataSet.shape[0]

    #得到分类数据组,用于计算向量
    sortedData=tile(inX,(dataSize,1))

    #计算向量
    VectorData=sortedData-dataSet

    #计算平方
    sqVectorData=VectorData**2

    #计算行数据的和,得到距离的平方
    SumsqVector=sqVectorData.sum(axis=1)

    #计算距离,即开方
    SqrtSum=SumsqVector**0.5

   # print(SqrtSum)

    #排序,这里的argsort返回的是索引值
    SortSqrtSum=SqrtSum.argsort()

   # print(SortSqrtSum)

    #选择前k个点
    classCount={}

    for i in range(k):

        #得到距离最近的数据的标签
        newLabel=labels[SortSqrtSum[i]]

        #将标签给classCount,并且计算相同代表距离的标签出现的次数
        classCount[newLabel]=classCount.get(newLabel,0)+1

    #itemgetter(1)指的是获取字典第一个域的值,即次数
    sortedclassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    #print(sortedclassCount)

    #将次数最多的返回
    return sortedclassCount[0][0]

def Irisatamatrix(filename):

    #打开数据包
    fr=open(filename)

    #读取数据
    ArrayLines=fr.readlines()

    print(ArrayLines)
    #获取数据数
    NumOfdata=len(ArrayLines)

    #建立一个数组读取数据
    OnlyData=zeros((NumOfdata,4))

    #建立标签,每个数据对应的标签
    OnlyDataLabels=[]

    index=0
    for line in ArrayLines:

        #去掉空白
        line=line.strip()
        #将数据存进来

        #print(line)

        DataOfLine=line.split(",")

       # print(DataOfLine)
        #将数据放进来

        OnlyData[index,:]=DataOfLine[0:4]


        index+=1

        # 将数据进行分类
        labels = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}

        # 无法将字符串直接转化为int,这里是在列表尾部增加,将数据的类型即标签存入classLabelVector
        OnlyDataLabels.append(labels[DataOfLine[-1]])


       # print(OnlyDataLabels)

    return OnlyData,OnlyDataLabels

#归一化特征值:(oldValue-min)/(max-min)
def autoNorm(dataSet):

    #取出每一列中最小的值
    minVals=dataSet.min(0)

    #取出每一列中的最大的值
    maxVals=dataSet.max(0)

    #取值范围
    ranges=maxVals-minVals

    #创建一个0矩阵,以dataset为模板
    normDataSet=zeros(shape(dataSet))

    #取出行的长度
    m=dataSet.shape[0]

    #向行的方向复制,并且减去最小值
    normDataSet=dataSet-tile(minVals,(m,1))

    #除去范围值
    normDataSet=normDataSet/tile(ranges,(m,1))

    return normDataSet,ranges,minVals

def main():
    #得到数据
    datingDataMat,datingLabels=Irisatamatrix('iris.data')

    #print(datingDataMat)
    #使用归一
    normMat,ranges,minVals=autoNorm(datingDataMat)

    #输出数据
    #print(datingDataMat)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(normMat[:,1],normMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))

    IrisTestData()

    plt.show()


def IrisTestData():
    # 选择前百分之十作为测试集
    hoRatio = 0.1

    # 打开数据集
    datingDataMat, datingLabels = Irisatamatrix('iris.data')

    # 将数据集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # m得到的是数据集的组数
    m = normMat.shape[0]

    # 得到前百分之十的组数的位置
    numTestVecs = int(m * hoRatio)

    # 错误的数量
    errorCount = 0.0

    # 前numTestVecs的数据进行测试,后numTestVecs的数据作为训练集
    for i in range(numTestVecs):

        # 返回每个数据的标签
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 60)

        # 将数据进行分类
        labels = {1: 'Iris-setosa', 2:'Iris-versicolor', 3: 'Iris-virginica'}

        # 输出数据的结果类型与真实类型
        print("the classifier came back with %s,the real answer is:%s" \
              % (labels[classifierResult], labels[datingLabels[i]]))

        # 如果对应类型错了,就加一
        if (classifierResult != datingLabels[i]): errorCount += 1.0

        # 输出错误的百分率
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))

main()