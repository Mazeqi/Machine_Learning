from TestTrees import calcShannonEnt
from TestTrees import chooseBestFeatureToSplit
from TestTrees import createTree
import TestTrees
import  TreePlot
def createDataSet():
    dataSet=[
               [1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']
            ]

    labels=['no surfacing','flippers']
    return dataSet,labels

fr=open('matpilb\lenses.txt')
lenses=[inst.strip().split('\t')for inst in fr.readlines()]
lensesLabels=['ages','prescript','astigmatic','tearRate']
print(lenses)
lensesTree=TestTrees.createTree(lenses,lensesLabels)
TreePlot.createPlot(lensesTree)