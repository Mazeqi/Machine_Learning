import matplotlib.pyplot as plt
from TestTrees import *

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#绘制指向
def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',
    xytext=centerPt,textcoords='axes fraction', va="center",ha="center",
                            bbox=nodeType,arrowprops=arrow_args)
'''
def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decision node ',(0.1,0.5),(0.1,0.3),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}},3:'maybe'}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#绘制树枝的权
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    #获得叶子的数目
    numLeafs=getNumLeafs(myTree)
    print("num of leafs : "+str(numLeafs))
    #获得树的深度
    depth=getTreeDepth(myTree)
    #树的key生成列表
    firstStr=list(myTree.keys())[0]
    #将根节点位置搞在一个元组里面

    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    print(cntrPt)

    #绘制树枝的特征值
    plotMidText(cntrPt,parentPt,nodeTxt)
    #绘制节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)

    #搞完根节点就搞子树啦
    secondDict=myTree[firstStr]
    #y值要向下减一
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD

    print(secondDict.keys())
    #遍历子树
    for key in secondDict.keys():
        #如果子树还有子树就递归
        if type(secondDict[key])==dict:
            plotTree(secondDict[key],cntrPt,str(key))
        #如果子树没有子树,就画出节点
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            print("the node xoff:%f"%plotTree.xOff)
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    #楼上的计算完要计算纵坐标
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    agprops=dict(xticks=[],yticks=[])
   # print(agprops)
    createPlot.ax1=plt.subplot(111,frameon=False,**agprops)

    #计算树的叶子节点数目,从而计算树的起始位置
    plotTree.totalW=float(getNumLeafs(inTree))
    #计算树的高度
    plotTree.totalD=float(getTreeDepth(inTree))
    #计算树的起始位置
    plotTree.xOff=-0.5/plotTree.totalW
   # print(plotTree.totalW)
    #print("plottree xoff:"+str(plotTree.xOff))
    #树的y起始位置
    plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)

    plt.show()



def main():
    myTree = retrieveTree(0)
    createPlot(myTree)
