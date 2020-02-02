from math import log
import operator

def calcShannonEnt(dataSet):

    #获得数据的数目
    numEntries=len(dataSet)

    #创建一个数据字典,每个数据作为key,每个数据各自出现的次数作为value
    labelCounts={}

    for featVec in dataSet:
        #获得每个数据
        currentLabel=featVec[-1]

        #判断字典里是否有该标签
        if currentLabel not in labelCounts.keys():
             labelCounts[currentLabel]=0

        #给出现的标签次数加一
        labelCounts[currentLabel]+=1

    #计算Ent
    shannonEnt=0.0
    for key in labelCounts:
        #每个数据出现的次数除以总的次数
        prob=float(labelCounts[key])/numEntries
        #累加
        shannonEnt-=prob*log(prob,2)

    return shannonEnt

def splitDataSet(dataSet,axis,value):

    #建一个新的列表
    retDataSet=[]

    #数据的第axis个特征进行分类
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#计算信息增益
def chooseBestFeatureToSplit(dataSet):
    #获得长度减一,用于对数据集的特征值进行处理
    numFeatures=len(dataSet[0])-1

    #获得数据集的分类比例
    baseEntropy=calcShannonEnt(dataSet)

    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        #dataSet的第example行的第i个元素加到featList中去
        featList=[example[i]for example in dataSet]

        #把第i列元素加载成集合,筛选掉重复的元素
        uniqueVals=set(featList)

        newEntropy=0.0

        #对集合中的每个特征进行一次计算Ent
        for value in uniqueVals:
            #将数据集按照每个value值进行分组
            subDataSet=splitDataSet(dataSet,i,value)
            #每个分好组的数据集求出其占总数据集的比例
            prob=len(subDataSet)/float(len(dataSet))
            #将每一组数据的Ent值乘以比例后加起来
            newEntropy+=prob*calcShannonEnt(subDataSet)

        #这里是信息增益,需要选取最大的
        infoGain=baseEntropy-newEntropy

        #如果这一次循环的信息增益比记录下来的大,就记录下这一列的列数,用于之后的决策树
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#这个函数是用在于当所有的属性已经被用光了,但是分类还是不彻底,就要让出现次数最多的列子来决定分类结果
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    #将数据集的标签生成一个列表
    classList=[example[-1]for example in dataSet]
    #当标签属性已经没有了
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #当数据集已经凉了
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #计算信息增益然后得到用分支的特征的列数
    bestFeat=chooseBestFeatureToSplit(dataSet)
    #选择用来分支的标签
    bestFeatLable=labels[bestFeat]
    #用标签生成字典
    myTree={bestFeatLable:{}}
    #删除已经用过的标签
    del(labels[bestFeat])
    #将用来分支的那一列的所有数据生成一个列表
    featValues=[example[bestFeat] for example in dataSet]
    #生成没有重复数据的集合
    uniqueVals=set(featValues)
    #print(uniqueVals)

    #利用对用来分支的那一列进行分类
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLable][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#获得叶子数
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key])==dict:
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

#获得树的长度
def getTreeDepth(myTree):
   maxDepth=0

   firstStr=list(myTree.keys())[0]
   secondDict=myTree[firstStr]
   for key in secondDict.keys():
       if type(secondDict[key])==dict:
           thisDepth=1+getTreeDepth(secondDict[key])
       else:
           thisDepth=1
       if thisDepth>maxDepth:
           maxDepth=thisDepth
   return maxDepth

def classify(inputTree,featLabels,testVec):
    #将树放在列表里
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    #返回firstStr的索引
    featIndex=featLabels.index(firstStr)
    print(featIndex)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key])==dict:
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]

    return classLabel

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}},3:'maybe'}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return  pickle.load(fr)