from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir


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

#打开文件,过滤数据
def file2matrix(filename):
    #打开文件
    fr=open(filename)

    #读取文件中的文本
    arrayOLines=fr.readlines()

    #得到文件数目
    numberOfLines=len(arrayOLines)

    #创建一个矩阵 n行3列
    returnMat=zeros((numberOfLines,3))

    #定义一个空矩阵
    classLabelVector=[]

    #下标为0
    index=0

    #读取每一行的数据
    for line in arrayOLines:
        #去掉空格
        line=line.strip()

        #以'\t'作为分隔符
        listFromLine=line.split('\t')

       # print(listFromLine)
        #第index行的数据存入returnMat,除了末尾的类型
        returnMat[index,:]=listFromLine[0:3]

       # print(returnMat)

        #将数据进行分类
        labels = {'1': 1, '2': 2, '3': 3}

        #无法将字符串直接转化为int,这里是在列表尾部增加,将数据的类型即标签存入classLabelVector
        classLabelVector.append(labels[listFromLine[-1]])

        #classLabelVector.append(listFromLine[-1])
        index+=1
    return returnMat,classLabelVector

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

#测试函数
def datingClassTest():
    #选择前百分之十作为测试集
    hoRatio=0.10

    #打开数据集
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')

    #将数据集归一化
    normMat,ranges,minVals=autoNorm(datingDataMat)

   #m得到的是数据集的组数
    m=normMat.shape[0]

    #得到前百分之十的组数的位置
    numTestVecs=int(m*hoRatio)

    #错误的数量
    errorCount=0.0

    #前numTestVecs的数据进行测试,后numTestVecs的数据作为训练集
    for i in range(numTestVecs):

        #返回每个数据的标签
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],7)

        #输出数据的结果类型与真实类型
        print("the classifier came back with %d,the real answer is:%d"\
              %(classifierResult,datingLabels[i]))

        #如果对应类型错了,就加一
        if(classifierResult!=datingLabels[i]):errorCount+=1.0

        #输出错误的百分率
    print("the total error rate is : %f" %(errorCount/float(numTestVecs)))

#预测函数
def classifyPerson():

    resultList=['not at all','in small doses','in large doses']

    #获取三个数值,用于计算结果
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))

    #打开训练集
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')

    #归一化数据集
    normMat,ranges,minVals=autoNorm(datingDataMat)

    # 输出数据
    # print(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normMat[:, 1], normMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))

    #将获得的三个数据存到列表
    inArr=array([ffMiles,percentTats,iceCream])

    #归一化得到的数据
    inArrto1=(inArr-minVals)/ranges

    #通过knn算法得到结果
    classiFierResult=classify0(inArrto1,normMat,datingLabels,3)

    #输出结果
    print("You will probably like this person:"+str(resultList[classiFierResult-1]))

    #将训练集的数据输出
    plt.show()

#这个函数将文件中的32*32的数据转化成1*1024
def img2vector(filename):

    #先建一个1*1024的列表
    returnVect=zeros((1,1024))

    #打开文件
    fr=open(filename)

    #这里一行一行读取,并一个数据一个数据存入
    for i in range(32):

        #读取一行
        lineStr=fr.readline()
        #print(lineStr)

        #将行中的数据存入列表
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])

    #返回列表
    return returnVect

#测试代码
def handwritingClassTest():

    #声明空标签
    hwLabels=[]

    #得到文件夹中的文件名所组成的列表
    trainingFileList=listdir('trainingDigits')

    #print(trainingFileList)

    #得到文件名的数目
    m=len(trainingFileList)

    #搞一个m行1024列的矩阵
    trainingMat=zeros((m,1024))


    for i in range(m):

        #对每一个txt文件中的数据进行读入
        fileNameStr=trainingFileList[i]

        #分割文件名
        fileStr=fileNameStr.split('.')[0]

        #再次分割
        classNumStr=int(fileStr.split('_')[0])

        #形成标签
        hwLabels.append(classNumStr)

        #将数据读取进来
        trainingMat[i,:]=img2vector('trainingDigits/%s'% fileNameStr)

    #读取测试数据的文件名
    testFileList=listdir('testDigits')

    #标记出错的数量
    errorCount=0.0

    #测试数据文件量
    mTest=len(testFileList)

    #对测试数据进行处理
    for i in range(mTest):

        #形成文件名标签
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])

        #得到数据
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)

        #对数据进行匹配并输出类型
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)

        #输出
        print("the classifier came back with:%d,the real answer is : %d" % (classifierResult,classNumStr))

        #计算数据出错量
        if(classifierResult!=classNumStr):errorCount+=1.0

    #输出
    print("\nthe total number of errors is : %d" % errorCount)
    print("\nthe total error rate is : %f "%(errorCount/float(mTest)))

