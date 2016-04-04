import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 상자 형태 정의 (상자 모양, 색)
leafNode = dict(boxstyle="round4", fc="0.8")  # 상자 형태 정의 (상자 모양, 색)
arrow_args = dict(arrowstyle="<-")  # 화살표 형태 정의 (화살표 모양)

# plotNode 함수
# nodeTxt:노드 텍스트
# centerPt: 자식 위치
# parentPt: 부모 위치
# nodeType: 노드 종류
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 주석 달기 (http://matplotlib.org/users/annotations_intro.html)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)
    # nodeTxt: text
    # xy: start point
    # xycoords: coordinate of xy (0,0 is lower left of axes and 1,1 is upper right)
    # xytext: text point
    # textcoords: coordinate of test (0,0 is lower left of axes and 1,1 is upper right)
    # bbox: box border
    # arrowprops: arrow propery


# createPlot 함수
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # clearfigure
    createPlot.ax1 = plt.subplot(111, frameon=False)  # createPlot은 전역변수
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# getNumLeafs 함수
# myTree: 트리 (dictionary)
# numLeafs: 트리의 노드 개수
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 노드가 Dictionary이면 의사결정 영역
            numLeafs += getNumLeafs(secondDict[key])  # 재귀
        else: numLeafs += 1  # 아니면 단말 영역
    return numLeafs

# getTreeDepth 함수
# myTree: 트리 (dictionary)
# numLeafs: 트리의 깊이
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 노드가 Dictionary이면 의사결정 영역
            thisDepth = 1 + getTreeDepth(secondDict[key])  # 재귀
        else: thisDepth = 1  # 아니면 단말 영역
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# retrieveTree 함수
# i: 트리 종류
# listOfTrees[i]: i번째 트리 예제
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

# plotMidText 함수
# cntrPt: 자식 노드 위치
# parentPt: 부모 노드 위치
# txtString: 문자열
def plotMidText(cntrPt, parentPt, txtString):
    # 자식 노드와 부모 노드 사이에 텍스트 플롯하기
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

    # plotTree 함수
# myTree: 트리 (dictionary)
# parentPt: 부모 노드 위치
# nodeTxt: 문자열
def plotTree(myTree, parentPt, nodeTxt):
    # 트리의 넓이와 높이를 계산함
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 자식 노드 값 플롯하기
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  # y offset 감소
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  # 노드가 Dictionary이면 의사결정 영역
            plotTree(secondDict[key],cntrPt,str(key))  # 재귀
        else:  # 아니면 단말 영역
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# createPlot 함수
# inTree: 트리 (dictionary)
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()