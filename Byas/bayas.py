'''
@author: Mazeqi
'''
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec



"""
    提取所有文档中的词条并进行去重
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
"""

def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 将集合并起来
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


"""
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含都为0的向量，该向量长度为len
    for word in inputSet:  # 遍历每个词
        if word in vocabList:  # 如果词条存在则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


'''
@param: trainMatrix为文档矩阵,一个0，1的向量矩阵
@param: trainCategory为每篇文档类别标签所构成的向量
'''
def trainNB0(trainMatrix, trainCategory):
    # 获取文档的数量
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 对两个标签建立向量
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # change to ones()

    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0

    # 计算出每个所占的比例
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现的概率
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect,pAbusive
"""
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
  """
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    #加载数据集
    listOPosts, listClasses = loadDataSet()
    #创建单词集合
    myVocabList = createVocabList(listOPosts)
    #计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #训练数据
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    #测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

'''
Desc:
    接受一个大字符串并将其解析为字符串列表
Args:
     bigString--大字符串
Returns:
       去掉少于 2 个字符的字符串，并将所有字符串转换为小写，返回字符串列表
'''
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


'''
    Desc:
        对贝叶斯垃圾邮件分类器进行自动化处理。
    Args:
        none
    Returns:
        对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
'''
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #切分,解析数据,并归类为1类别
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        #切分,解析数据,并归类为0类别
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    #创建词汇表
    vocabList = createVocabList(docList)  # create vocabulary

    trainingSet = list(range(50))
    testSet = []  # create test set

    #随机取十个邮件来测试
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:  # classify the remaining items

        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])

        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])

    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}

    #遍历词汇表中的每一个词
    for token in vocabList:
        #统计每个词在文本中出现的次数
        freqDict[token] = fullText.count(token)
    # 根据每个词出现的次数从高到底对字典进行排序
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)

    #返回出现次数最高的三十个单词
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minLen):
        #每次访问一条Rss源,除了第一句其他都一样
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words

    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(2 * minLen));
    testSet = []  # create test set

    for i in list(range(10)):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

#最具表征性的词汇显示函数
'''
 函数使用两个rss源作为输入，然后训练并测试朴素贝叶斯分类器，
 返回使用的概率值。然后创建两个列表用于元组的存储，与之前返
 回排名最高的 X 个单词不同，这里可以返回大于某个阈值的所有词，
 这些元组会按照它们的条件概率进行排序。
'''
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)

    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")

    for item in sortedSF:
        print(item[0])

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")

    for item in sortedNY:
        print(item[0])