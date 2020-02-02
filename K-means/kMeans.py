from numpy import *


# 同以往的操作,将数据封装到矩阵
def loadDataSet(filename):
    # 将数据存到列表中
    dataMat = []
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split("\t")
        # p3中返回的都是迭代器,需要重新搞成list
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat


# 计算欧式距离
def disEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


# 构建簇的质心
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    # 这里是一列一列算,最大到最小之间选出质心
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids


def kMeans(dataSet, k, distMeas = disEclud, createCent = randCent):

    # 获得数据的组数
    dataShape = shape(dataSet)[0]

    # 生成一个m行,2列的矩阵,用来存每个数据到簇的距离,以及对应簇的类型
    clusterAssment = mat(zeros((dataShape,2)))

    # 获得随机质心
    centroids = createCent(dataSet,k)

    # 判断是否继续
    clusterChanged=True

    while clusterChanged:

        clusterChanged = False

        # 将每组数据跟k个求距离
        for i in range(dataShape):

            # 先将最小距离设为无穷
            minDist = inf

            # 索引是-1
            minIndex = -1

            # 这里对每个数据进行分类
            for j in range(k):

                # 计算距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])

                # 判断是不是最小距离
                if distJI < minDist:
                    minDist = distJI

                    # 记住对应簇的类型
                    minIndex = j

            # 如果与某个数据对应的最近距离的点的下标出现了变化,则更改
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist**2

        # print(centroids)
        # 更新簇
        for cent in range(k):

            # https: // blog.csdn.net / xinjieyuan / article / details / 81477120
            # nonzero返回使括号中为True的下标
            cs = nonzero(clusterAssment[:, 0].A == cent)[0]

            # print(cs)
            ptsInClust = dataSet[cs]

            # 对每一行进行更新，这里是更新簇
            centroids[cent, :] = mean(ptsInClust, axis=0)

        # print(clusterAssment)
    return centroids, clusterAssment


# 二分聚类
def biKmeans(dataSet,k,distMeas=disEclud):

    # 建一个矩阵,用来存放每个数据对应的簇以及其到簇心的距离
    dataRow = shape(dataSet)[0]

    clusterAssment = mat(zeros((dataRow,2)))

    # 建立第一个簇心
    centroidO=mean(dataSet,axis=0).tolist()[0]

    # print(centroidO)
    # 创建一个带有质心的 [列表]，因为后面还会添加至k个质心
    centList = [centroidO]

    # print(centList)
    for j in range(dataRow):

        # 计算每个数据到簇心的距离
        clusterAssment[j, 1] = distMeas(mat(centroidO), dataSet[j, :])**2

    # print(clusterAssment[:,0].A)
    # 进行二分聚类,注意:二分这里只有两种即 0 跟 1
    while (len(centList)<k):
        lowestSSE=inf
        for i in range(len(centList)):
            # 获得属于该质点的所有样本数据
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]

            # 返回中心点信息、该数据集聚类信息
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)

            # 这是划分数据的SSE    加上未划分的 作为本次划分的总误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])

            print("sseSplit,and notSplit",sseSplit,sseNotSplit)

            # 将划分与未划分的SSE求和与最小SSE相比较 确定是否划分
            if(sseSplit+sseNotSplit)<lowestSSE:
                # 得出当前最适合做划分的中心点

                bestCentToSplit = i

                # 划分后的两个新中心点
                bestNewCents = centroidMat

                # 划分点的聚类信息
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit+sseNotSplit

        # 由于是二分，所有只有0，1两个簇编号，将属于1的所属信息转为下一个中心点
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)

        # 将属于0的所属信息替换用来聚类的中心点
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit

        print('the bestCentToSpilt is : ',bestCentToSplit)
        print('the len of bestClustAss is :',len(bestClustAss))

        # 与上面两条替换信息相类似，这里是替换中心点信息，上面是替换数据点所属信息
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])

        print(bestNewCents)
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0], :] = bestClustAss

    # print(clusterAssment)
    return mat(centList), clusterAssment