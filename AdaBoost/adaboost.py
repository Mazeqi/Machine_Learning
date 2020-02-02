from numpy import *
def loadSimpData():
    datMat = matrix\
    (
      [
         [1.,2.1],
         [2.,1.1],
         [1.3,1.],
         [1.,1.],
         [2.,1.]
      ]
    )
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]

    return datMat,classLabels

'''
  dataMatrix:数据集
  dimen：数据集列数
  threshVal：阈值
  threshIneq：比较方式 IT,gt
  
  return : retArray:分类结果
'''


# 单层决策树的阈值过滤函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    # 新建一个数组用于存放分类结果，初始化都为1
    retArray = ones((shape(dataMatrix)[0], 1))

    # It:小于，gt：大于，根据阈值进行分类，并将分类结果存储到retArray
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


"""
    Function：   找到最低错误率的单层决策树

    Input：     dataArr：数据集
                classLabels：数据标签
                D：权重向量

    Output：    bestStump：分类结果
                minError：最小错误率
                bestClasEst：最佳单层决策树
                
#构建单层分类器
#单层分类器是基于最小加权分类错误率的树桩
#伪代码
#将最小错误率minError设为+∞
#对数据集中的每个特征(第一层特征)：
    #对每个步长(第二层特征)：
        #对每个不等号(第三层特征)：
            #建立一颗单层决策树并利用加权数据集对它进行测试
            #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
#返回最佳单层决策树
"""


# 构建基学习器
def buildStump(dataArr, classLabels, D):

    # 初始化数据集和数据标签
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T

    # 获取行列值
    m, n = shape(dataMatrix)

    # 初始化步数，用于在特征的所有可能值上进行遍历
    numSteps = 10.0

    # 初始化字典，用于存储给定权重向量D时所得到的最佳层决策树的相关信息
    bestStump = {}

    # 初始化类别估计值
    bestClasEst = mat(zeros((m, 1)))

    # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
    minError = inf  # init error sum, to +infinity

    # 遍历数据集的每个特征
    for i in range(n):  # loop over all dimensions

        # 获取数据集的最大最小值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()

        # 根据步数求得步长
        stepSize = (rangeMax - rangeMin) / numSteps

        # 遍历每个步长
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension

            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']:  # go over less than and greater than

                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal = (rangeMin + float(j) * stepSize)

                # 通过阈值比较对数据进行分类
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan

                # 初始化错误计数向量
                errArr = mat(ones((m, 1)))

                # 如果预测结果和标签相同，则相应位置0
                errArr[predictedVals == labelMat] = 0

                # 计算权值误差，这就是AdaBoost和分类器交互的地方
                weightedError = D.T * errArr  # calc total error multiplied by D

                # print("split: dim %d, thresh %.2f, thresh ineqal:\
                #      %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))

                # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    # 返回最佳单层决策树，最小错误率，类别估计值
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):

    """
    Function：   找到最低错误率的单层决策树

    Input：      dataArr：数据集
                 classLabels：数据标签
                 numIt：迭代次数

    Output： weakClassArr：单层决策树列表
             aggClassEst：类别估计值
             
#完整AdaBoost算法实现
#算法实现伪代码
#对每次迭代：
    #利用buildStump()函数找到最佳的单层决策树
    #将最佳单层决策树加入到单层决策树数组
    #计算alpha
    #计算新的权重向量D
    #更新累计类别估计值
    #如果错误率为等于0.0，退出循环

    """
    # 初始化列表，用来存放单层决策树的信息
    weakClassArr = []

    # 获取数据集行数
    m = shape(dataArr)[0]

    # 初始化向量D每个值均为1/m，D包含每个数据点的权重
    D = mat(ones((m,1))/m)

    # 初始化列向量，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))

    # 开始迭代
    for i in range(numIt):

        # 利用buildStump()函数找到最佳的单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)

        # print("D: ", D.T)
        # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))

        # 保存alpha的值
        bestStump['alpha'] = alpha

        # 填入数据到列表
        weakClassArr.append(bestStump)

        # print("classEst: ", classEst.T)
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)

        # 更新权值向量
        D = multiply(D, exp(expon))
        D = D / D.sum()

        # 累加类别估计值
        aggClassEst += alpha * classEst

        # print("aggClassEst: ", aggClassEst.T)
        # 计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)

        # 如果总错误率为0则跳出循环
        if errorRate == 0.0:
            break

    # 返回若分类器组成的列表
    # return weakClassArr
    return weakClassArr, aggClassEst


def adaClassify(datToClass , classifierArr):

    """
    Function：   AdaBoost分类函数

    Input：      datToClass：待分类样例
                classifierArr：多个弱分类器组成的数组

    Output： sign(aggClassEst)：分类结果
    """

    # 初始化数据集
    dataMatrix = mat(datToClass)

    # 获得待分类样例个数
    m = shape(dataMatrix)[0]

    # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))

    # 遍历每个弱分类器
    for i in range(len(classifierArr)):

        # 基于stumpClassify得到类别估计值，每个弱分类器对测试数据进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq'])

        # 累加类别估计值
        aggClassEst += classifierArr[i]['alpha']*classEst

        # 打印aggClassEst，以便我们了解其变化情况
        # print(aggClassEst)

    # 返回分类结果，aggClassEst大于0则返回+1，否则返回-1
    return sign(aggClassEst)


def loadDataSet(fileName):

    numFeat=len(open(fileName).readline().split('\t'))

    dataMat=[]
    labelMat=[]
    fr=open(fileName)

    for line in fr.readlines():

        lineArr=[]

        curLine=line.strip().split('\t')

        for i in range(numFeat-1):

            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)

        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt

    # 保留绘制光标的位置
    cur = (1.0, 1.0)  # cursor

    # 计算AUC的值
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)

    # 获取排序索引
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    # loop through all the values, drawing a line segment at each poin
    # 画图
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)

