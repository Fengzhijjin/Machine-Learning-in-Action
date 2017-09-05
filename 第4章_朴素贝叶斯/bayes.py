#coding:utf-8

from numpy import *

def loadDataSet():    #创建实验样本，输出样本集和标签集
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0, 1, 0, 1, 0, 1]    #1 代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):    #创建数据集中中不重复词表
    vocabSet = set([])    #创建空集
    for document in dataSet:    #循环遍历获取数据集元素
        vocabSet = vocabSet | set(document)    #创建并集
    return list(vocabSet)    #输出列表

#vacaList  不重复词表
#inputSet  测试列表或文档
def setOfWords2Vec(vocabList, inputSet):    #测试不重复词表中单词在测试文档中是否出现
    returnVec = [0]*len(vocabList)    #创建记录向量
    for word in inputSet:    #循环遍历测试集
        if word in vocabList:    #检测不重复词汇表，若出现则记录向量位置+1，若未出现则打印未出现
            returnVec[vocabList.index(word)] = 1
        else:
            print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec    #返回记录向量

#trainMatrix  文档不重复词表出现次数矩阵
#trainCategory  类别标签向量
def trainNB0(trainMatrix,trainCategory):    #朴素贝叶斯分类器训练函数
    numTrainDocs = len(trainMatrix)    #计算测试数量
    numWords = len(trainMatrix[0])    #不重复词表中单词数量
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #计算侮辱性文档概率
    #以下四行初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):    #遍历测试文档不重复词表出现次数矩阵，统计侮辱性文章和非侮辱性文章出现次数及单词出现次数
        if trainCategory[i] == 1:    #侮辱性文章次数统计
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:    #非侮辱性文章次数统计
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)    #计算侮辱性文章除以该文章总次数概率
    p0Vect = log(p0Num/p0Denom)    #计算非侮辱性文章除以该文章总次数概率
    return p0Vect,p1Vect,pAbusive

#vec2Classify  检测向量的不重复词表出现向量
#p0Vec,p1Vec,pClass1  函数trainNB0计算得到的三个概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):    #朴素贝叶斯分类判定函数
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #计算侮辱性文章概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)    #计算非侮辱性文章概率
    if p1 > p0:    #判断输出
        return 1
    else:
        return 0

def testingNB():    #便利函数，对操作进行封装
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))    #以上为训练算法
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))    #生成测试向量中不重复单词出现次数向量
    print (testEntry,'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'calssified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#vacaList  不重复词表
#inputSet  测试列表或文档
def bagOfWords2VecMN(vocabList, inputSet):    #朴素贝叶斯词袋模型，统计不重复词表中单词在测试文档中出现次数
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#实例：使用朴素贝叶斯过滤垃圾邮件

def textParse(bigString):    #文件解析函数，接受一个大字符串并将其解析为字符串列表
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    #去掉少于2个字符的字符串，并将所有字符串转换成小写

def spamTest():
    docList = []    #文档列表
    classList = []    #标签列表
    fullText = []    #单词列表
    for i in range(1, 26):    #读取文档信息
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)    #添加标签spam
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)    #添加标签ham
    vocabList = createVocabList(docList)    #创建不重复词表
    trainingSet = range(50)
    testSet = []    #测试集
    for i in range(10):    #随机选取测试文档，构建测试集，并从训练集中删除已选择文档
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []    #训练集
    trainClasses = []    #训练集标签
    for docIndex in trainingSet:    #循环遍历训练集，训练朴素贝叶斯算法
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0    #记录朴素贝叶斯算法判断错误次数
    for docIndex in testSet:    #循环遍历测试集，应用贝叶斯算法判断
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    #若判断错误，错误记录加1并打印
            errorCount += 1
            print ("classification error", docList[docIndex])
    print ('the error rate is: ', float(errorCount) / len(testSet))    #打印错误率

