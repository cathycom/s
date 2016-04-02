from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) # trainingSet
    labels = ['A','A','B','B'] # labels
    return group, labels

#  KNN 분류 알고리즘
#  - inX : input data X
#  - dataSet : training data set
#  - labels : class label
#  - k : the number of similar examples in S
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # dataSet 행렬의 row 갯수
    # 유클리디안 거리 계산
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # dataSet과 inX의 차를 계산
    sqDiffMat = diffMat**2 # 제곱 연산
    sqDistances = sqDiffMat.sum(axis=1) # 두 데이터의 합을 계산
    distances = sqDistances**0.5 # 루트 연산
    # 오름차순 정렬
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # a = a + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True) # 내림차순 정렬
    return sortedClassCount[0][0]




def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #classLabelVector.append(listFromLine[-1])
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

# normalization : newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 1x3
    #print minVals
    maxVals = dataSet.max(0)  # 1x3
    #print maxVals
    ranges = maxVals - minVals  # 1x3
    #print ranges
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] # length
    normDataSet = dataSet - tile(minVals, (m, 1))  # 1x3 행렬(minVals)을 1000x3 행렬(dataSet)로 변환하기 위해 tile을 사용함
    normDataSet = normDataSet / tile(ranges, (m, 1))  # x3 행렬(ranges)을 1000x3 행렬(normDataSet)로 변환하기 위해 tile을 사용함
    return normDataSet, ranges, minVals  # 추후 testSet 데이터도 정규화시키기 위해 ranges, minVals 값도 반환함

# normalization : newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 1x3
    #print minVals
    maxVals = dataSet.max(0)  # 1x3
    #print maxVals
    ranges = maxVals - minVals  # 1x3
    #print ranges
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] # length
    normDataSet = dataSet - tile(minVals, (m, 1))  # 1x3 행렬(minVals)을 1000x3 행렬(dataSet)로 변환하기 위해 tile을 사용함
    normDataSet = normDataSet / tile(ranges, (m, 1))  # x3 행렬(ranges)을 1000x3 행렬(normDataSet)로 변환하기 위해 tile을 사용함
    return normDataSet, ranges, minVals  # 추후 testSet 데이터도 정규화시키기 위해 ranges, minVals 값도 반환함

# training set의 일부분을 (10%) test set으로 활용하여 검사함
def datingClassTest():
    hoRatio = 0.10 # test set 비율 (10%)
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt') # read data
    normMat, ranges, minVals = autoNorm(datingDataMat) # normalization
    m = normMat.shape[0] # length of training set
    numTestVecs = int(m * hoRatio) # length of test set
    errorCount = 0.0
    for i in range(numTestVecs):
        # inX : normMat[i, :] (test set)
        # dataSet : normMat[numTestVecs:m, :] (test set을 제외한 training set)
        # labels : dataingLabels[numTestVecs:m, 3] (test set을 제외한 labels)
        # k : 3
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3) # classify
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('data/datingTestSet2.txt') # read data
    normMat, ranges, minVals = autoNorm(datingDataMat) # normalization
    inArr = array([ffMiles, percentTats, iceCream]) # inX
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3) # classify
    print "You will probably like this person: ", resultList[int(classifierResult) - 1]




def img2vector(filename):
    returnVect = zeros((1,1024)) # 32 x 32 = 1024
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []  # labels
    trainingFileList = listdir('data/trainingDigits')  # load the training set
    m = len(trainingFileList)  # 파일 리스트 갯수
    trainingMat = zeros((m, 1024))  # data set
    for i in range(m):
        # 파일명에서 분류 번호를 처리함
        fileNameStr = trainingFileList[i]  # 0_13.txt
        fileStr = fileNameStr.split('.')[0]  # 0_13
        classNumStr = int(fileStr.split('_')[0])  # 0
        hwLabels.append(classNumStr)  # 중복은 추가 안됨
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)  # trainingDigits/0_13.txt

    testFileList = listdir('data/testDigits')  # load the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 0_13.txt
        #fileStr = fileNameStr.split('.')[0]  # 0_13
        classNumStr = int(fileNameStr.split('_')[0])  # 0
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        # classify
        classifierResult = classify0(vectorUnderTest,
                                    trainingMat, hwLabels, 3)
        #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))