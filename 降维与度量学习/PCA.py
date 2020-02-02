from numpy import *


def loadDataSet(fileName, delim='\t'):

    fr = open(fileName)

    stringArr = [line.strip().split(delim) for line in fr.readlines()]

    datArr = [list(map(float,line)) for line in stringArr]

    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    # 对所有样本进行中心化
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean

    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)

    # 对协方差进行特征值分解，前面是特征值后面是特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))

    # 返回最大的d个特征值所对应的特征向量
    eigValInd = argsort(eigVals)            # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       # reorganize eig vects largest to smallest

    # 用特征向量转换到该维度
    lowDDataMat = meanRemoved * redEigVects   # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # WT*W=1

    return lowDDataMat, reconMat


def replaceNanWithMean():

    datMat = loadDataSet('secom.data', ' ')

    numFeat = shape(datMat)[1]

    for i in range(numFeat):

        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])  # values that are not NaN (a number)

        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  # set NaN values to mean

    return datMat