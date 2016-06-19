from numpy import *
from konlpy.tag import Twitter
from svmutil import *

def loadDataSet(file):
    with open(file,'r',encoding="utf8") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] #header 제외 : 텍스트 맨 위 id, document, label
        dataSet = [inst[1] for inst in data] # document
        labels = [int(inst[2]) for inst in data] # label
    return dataSet, labels

train_dataSet, train_labels = loadDataSet('ratings_test.txt')

tagger = Twitter()
def tokenize(doc):
    return tagger.nouns(doc)

train_doc = [tokenize(row) for row in train_dataSet]

# function createVocabList: 유일한 단어 목록 생성
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

# function setOfWords2Vec: 주어진 문서 내에 어휘 목록에 있는 단어가 존재하는지 아닌지를 확인
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

# function trainNB0: 나이브 베이스 분류기 훈련
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom=2.0; p1Denom=2.0

    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += (trainMatrix[i])
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += (trainMatrix[i])
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec)
    p0 = sum(vec2Classify * p0Vec)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet('ratings_test.txt')
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # training
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(train_labels))
    # test
    errorCount = 0
    for testIndex in range(0, len(trainMat)):
        thisDoc = array(setOfWords2Vec(myVocabList, listOPosts[testIndex]))
        if classifyNB(thisDoc,p0V,p1V,pAb) != train_labels[testIndex]:
            errorCount += 1
    print ('the error rate is:', float(errorCount)/len(listOPosts)*100)
    testEntry = ['별로다','노잼','돈아깝다']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['최고','재밌다','감동']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def trainSVM(trainMatrix, trainCategory):
    svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

    prob = svm_problem(trainCategory, trainMatrix)
    param = svm_parameter()
    param.kernel_type = LINEAR
    param.C = 10

    model = svm_train(prob, param)
    return model

def testingSVM(trainMat, listClasses):
    listClasses = map(int, listClasses)
    model = trainSVM(trainMat, listClasses) # training svm
    svm_predict(listClasses, trainMat, model)