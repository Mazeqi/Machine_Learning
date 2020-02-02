import bayas
from numpy import *

if __name__ == '__main__':
    listOPosts,listClasses=bayas.loadDataSet()

    print(listClasses)
    myVocabList=bayas.createVocabList(listOPosts)

    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bayas.setOfWords2Vec(myVocabList,postinDoc))

    print(trainMat)

    p0v,p1v,pAb=bayas.trainNBO(trainMat,listClasses)

    print(p0v)

    print(p1v)

    print(pAb)



'''
listOPosts,istClasses=bayas.loadDataSet()
myVocabList=bayas.createVocabList(listOPosts)
print(myVocabList)
print(bayas.setOfWords2Vec(myVocabList,listOPosts[3]))



#print(myVocabList)
'''