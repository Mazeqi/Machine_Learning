from numpy import *

def LoadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        #去掉首尾的空格，然后以中间的空格进行切分
        lineArr=line.strip().split()

        #处理数据
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])

        #处理标签
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


'''
sigmoid函数的输入记为z,即z=w0x0+w1x1+w2x2+...+wnxn，
如果用向量表示即为z=wTx，它表示将这两个数值向量对
应元素相乘然后累加起来。

梯度上升法(等同于我们熟知的梯度下降法，前者是寻找最大值，
后者寻找最小值)，它的基本思想是：要找到某函数的最大值，
最好的方法就是沿着该函数的梯度方向搜寻。如果函数为f，
梯度记为D，a为步长，那么梯度上升法的迭代公式为：w：w+a*Dwf(w)。a指阿法
该公式停止的条件是迭代次数达到某个指定值或者算法达到某个允许的误差范围。
我们熟知的梯度下降法迭代公式为：w：w-a*Dwf(w)
'''
# 梯度上升算法
def gradAscent (dataMatIn,classLabels):
    # 矩阵
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()

    # 获取矩阵的rows 跟cols
    m,n=shape(dataMatrix)

    # alpha是学习率
    alpha=0.001

    # 最大迭代次数
    maxCycles = 500

    # 最开始的权重设为1
    weights=ones((n,1))

    for k in range(maxCycles):
        # m行n列乘以 n行1列形成一个m行1列的矩阵，这里是预测属于哪个类别
        h = sigmoid(dataMatrix*weights)

        # https: // blog.csdn.net / ACdreamers / article / details / 44657979
        # 这里的error用了极大似然估计，匹配了sigmoid的概率模型
        # 错误率，用于学下一次权重
        error=(labelMat-h)

        # error是m行一列，dataMatrix是m行n列
        weights = weights+alpha*dataMatrix.transpose()*error

    return weights

def plotBestFit(weights):
    import  matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    # 得到数据的行数
    n=shape(dataArr)[0]

    xcord1=[]
    xcord2=[]
    ycord1=[]
    ycord2=[]

    # 对数据进行分类
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2, ycord2,s=30,c='green')

    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def socGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    # 每一次只对一行进行梯度上升
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]

    return weights

def socGradAscent1(dataMatrix,classLabels,numIter=500):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):

        dataIndex=range(m)

        for i in range(m):
            # alpha动态变化是为了保证在多次迭代之后新数据仍然具有一定的影响力
            # 如果处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数
            alpha=4/(1.0+j+i)+0.0001

            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])

    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    # 数据处理
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(21):
            lineArr.append(float(currLine[i]))

        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = socGradAscent1(array(trainingSet), trainingLabels, 400)

    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():

        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []

        for i in range(21):
            lineArr.append(float(currLine[i]))

        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec)

    print("the error rate of this test is: %f" % errorRate)

    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

