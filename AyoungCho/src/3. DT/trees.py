from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 가능한 모든 분류 하옴겡 대한 딕셔너리 생성
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 섀넌 엔트로피 계산
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 밑수가 2인 로그
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 분할 리스트 생성
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 분류 항목 표시에 대해 중복이 없는 리스트 생성
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 각각의 분할을 위해 엔트로피 계산
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):  # 가장 큰 정보 이득 찾기
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # count 횟수로 내림차순 정렬
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 모든 분류 항목이 같을 때 멈춤
        return classList[0]
    if len(dataSet[0]) == 1:  # 속성이 더 이상 없을 때 가장 많은 수를 반환함
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 최적 속성 찾기
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])  # 유일한 값의 리스트를 구함
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# classify 함수
# inputTree: 트리 (dictionary)
# featLabels: 라벨
# testVec: testSet
# classLabel: 분류된 라벨
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 색인을 위한 분류 항목 표시 문자열 변환
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

# storeTree 함수
# inputTree: 저장할 트리 객체
# filename: 저장할 파일 이름
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)  # 객체 저장
    fw.close()

# grabTree 함수
# filename: 로드할 파일 이름ㄹ
# pickle.load(fr): 로드된 트리 객체
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)  # 객체 로드

def lenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    treePlotter.createPlot(lensesTree)