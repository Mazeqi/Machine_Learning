import random
from numpy import *
#数据清理
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


#i是alpha的下标,m是所有alpha的数目
def selectJrand(i,m):
    j=i
    while(j==i):
        #生成0到m之间的实数,然后把它转成整数
        j=int(random.uniform(0,m))
    return j
#用于调整大于H或小于L的alpha的值

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

#param toler:容错率
#maxter:迭代的最大次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()

    #WX+b中的b
    b=0
    m,n=shape(dataMatrix)
    #初始化阿法向量,全部都为0
    alphas=mat(zeros((m,1)))
    iter=0

    # 当迭代次数小于最大迭代次数
    while(iter<maxIter):
        #用于记录alpha是否已经优化,优化就是求极值,如果已经优化,则加一
        alphaPairsChanged=0

        #遍历样本
        for i in range(m):
            # 样本i的预测值,multiply:对应元素相乘
            fXi=float(multiply(alphas,labelMat).T*\
                      (dataMatrix*dataMatrix[i,:].T))+b
            #样本的误差
            Ei=fXi-float(labelMat[i])

            '''
                 labelMat[i] = yi
                 Ei = g(xi) - yi
                 g(xi) = w*xi + b
                 所以：labelMat[i]*Ei = yi*(g(xi) - yi) = yi*g(xi) - 1
                 yi*g(xi) = labelMat[i]*Ei + 1
                 0 <= alpha_i <= C
                 当alpha_i = 0或C时，后面调整时将受门限函数制约，调整为L或H
                 加入罚参数C的KKT条件为 alpha_i*(yi*g(xi) - 1) = 0
                 所以选KKT条件 ：0 < alpha_i < C，等价于 yi*g(xi) = 1 ，等价于labelMat[i]*Ei + 1 = 1 即 labelMat[i]*Ei = 0

                 **KKT条件比较苛刻，需要一个容忍值toler， 那么上式 labelMat[i]*Ei = 0 的条件放宽为 abs(labelMat[i]*Ei) < toler

                 所以KKT条件为： abs(labelMat[i]*Ei) < toler 且 0 < alpha_i < C
                 至于if conditions 是否等价于 不满足KKT的条件， 目前没想明白
                 原论文给出的判断条件 ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
                 判断条件可以结合原问题的拉格朗日函数来看L(w,b,alpha) = 1/2 * ||w||**2 -∑ alpha * (yi*g(xi)-1)
            '''

            if((labelMat[i]*Ei<-toler)and(alphas[i]<C))or\
                    ((labelMat[i]*Ei>toler)and
                         (alphas[i]>0)):

                #随机选择第二个阿法值
                j=selectJrand(i,m)
                fXj=float(multiply(alphas,labelMat).T*
                          (dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                #两个图的两种情况,当选的阿法的y值相等或者不相等时的两种情况
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                #上下限相等就不需要再继续了
                if L==H:
                    print("L==H")
                    continue

                # K11+K22-2K12，K11 = X1*X1.T，K12 = X1*X2.T, 这里是原结论的负数形式，那么理论上eta应该小于0
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T

                if eta>=0:print("eta>=0");continue
                #eta是负的,所以要减去
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                # 门限函数阻止alpha_2的修改量过大(alpha_2剪辑后的解)
                alphas[j]=clipAlpha(alphas[j],H,L)

                #新加的一步优化:如果更新步长太小,则放弃
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue


                # 书上原公式：alpha_1_new = alpha_1_old + y1*y2(alpha_2_old - alpha_2_new)

                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                # b1_new = b1_old - E1 - y1* K11 *(alpha_1_new - alpha_1_old) - y2* K21 *(alpha_2_new - alpha_2_old)
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                        dataMatrix[i,:]*dataMatrix[i,:].T-\
                        labelMat[j]*(alphas[j]-alphaJold)*\
                        dataMatrix[i,:]*dataMatrix[j,:].T

                # b2_new = b2_old - E2 - y1* K12 *(alpha_1_new - alpha_1_old) - y2* K22 *(alpha_1_new - alpha_1_old)
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                        dataMatrix[i,:]*dataMatrix[j,:].T-\
                        labelMat[j]*(alphas[j]-alphaJold)*\
                        dataMatrix[j,:]*dataMatrix[j,:].T

                # 如果 alpha_1_new，alpha_2_new 同时在(0,C),则 b1_new = b2_new，此时直接赋值,否则取中点
                if(0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                #记录迭代完成
                alphaPairsChanged+=1
                print("iter : %d i:%d,pairs changed%d"%\
                      (iter,i,alphaPairsChanged))
        #当上面的代码把阿法优化到不能再优化的时候,程序会自动退出
        if(alphaPairsChanged==0):iter+=1
        else:iter=0
        print("iteration:%d"%iter)
    return b,


class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler,kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        #误差E缓存, 第一列是是否有效的标志位，第二列才是实际值,把误差缓存存起来的好处是能重复利用,节省每一次计算的时间
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

#计算误差Ei
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T *oS.K[:,k]  + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#第二个变量的选择,选择那个使Ei-Ej的绝对值最大的
def selectJK(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    #储存最大的|Ei-Ej|
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E

    # 返回第一列非0的坐标值，即Ei的i
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej


def updateEkK(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1, Ek]

#这里的过程都是同上面的simple一样,只不过多了个跟新
def innerLK(i, oS):
    Ei = calcEkK(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJK(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEkK(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print( "j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # update i by the same amount as j
        updateEkK(oS, i)  # added this for the Ecache
        #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                         oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
                         oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold)
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoPK(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):  # full Platt SMO
    #创建一个对象
    oS = optStructK(mat(dataMatIn), mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            #遍历所有的值
            for i in range(oS.m):
                #阿法的值已经不能再优化的时候就加一
                alphaPairsChanged += innerLK(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

#得到阿法值之后就可以计算w了
def clacWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem --That Kernel is not recognized')
    return K

def testRbf(k1=1):
    #读取数据
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    #调用smo函数,得到b跟alphas的值
    b,alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000,('rbf', k1)) #C=200 important
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    #过滤掉为0的值
    svInd=nonzero(alphas.A>0)[0]
    #阿法非0就是支撑向量
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        #将支撑向量与数据集传到核函数
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print( "the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#处理数据
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    #打开文件,把文件中所有的文件名生成一个列表
    trainingFileList = listdir(dirName)
    #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #处理每个文件名
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('digits//trainingDigits')
    b,alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr);
    labelMat = mat(labelArr).transpose()

    #取出支撑向量,以及支撑向量的标签
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];

    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))

    dataArr,labelArr = loadImages('digits//testDigits')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m) )