#coding:utf-8

'''
创建分支的伪代码函数createBranch（）：
    If so return 类标签;
    Else
        寻找划分数据集的最好特征
        划分数据集
        创建分支节点
            for 每个划分的子集
                调用函数createBranch并增加返回结果到分支节点中
        return 分支节点
'''

from math import log
import operator

def calcShannonEnt(dataSet):    #计算给定数据集的熵
    numEntries = len(dataSet)    #计算给定数据集的长度
    labelCounts = {}    #创建一个空字典
    for featVec in dataSet:    #找出给定数据集的分类类别及类别数
        currentLabel = featVec[-1]    #遍历每组数据的类别
        if currentLabel not in labelCounts.keys():    #如果字典中没有此类别则初始化为0
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    #如果字典中有此类别则其值加1
    shannonEnt = 0.0    #初始化一个浮点数 熵
    for key in labelCounts:    #遍历字典labelCounts   进行熵值计算
        prob = float(labelCounts[key])/numEntries    #求每种分类出现的概率
        shannonEnt -= prob * log(prob,2)    #求熵计算
    return shannonEnt    #返回所给数据集的熵值

def createDataSet():    #输入测试集与分类属性
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]    #测试数据集
    labels = ['no surfacing','flippers']    #分类属性集
    return dataSet, labels

#dataSet  待划分的数据集
#axis  划分数据集的特征即属性列号
#value  特征的返回值
def splitDataSet(dataSet, axis, value):    #去掉数据集中已经划分的属性,返回已去除属性并符合要求的数据集
    retDataSet = []    #创建一个空列表
    for featVec in dataSet:    #遍历数据集中每一组数据
        if featVec[axis] == value:    #判断数据中属性是否相符
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])    #去除数据集集中axis属性
            retDataSet.append(reducedFeatVec)
    return retDataSet    #返回数据集中已去除划分属性的新集合

def chooseBestFeatureToSplit(dataSet):    #ID3决策数算法计算选择出数据集的最好划分方式
    numFeatures = len(dataSet[0]) - 1    #计算数据集中的特征数量
    baseEntropy = calcShannonEnt(dataSet)    #计算原数据集熵
    bestInfoGain = 0.0    #初始化为0的信息增益
    bestFeature = -1
    for i in range(numFeatures):    #遍历每一种特征的信息增益，选出最好的数据划分特征
        featList = [example[i] for example in dataSet]    #遍历获取第i个属性的全部取值列表
        uniqueVals = set(featList)    #从列表中创建集合，得到不重复的所有可能取值
        newEntropy = 0.0    #初始化为0的新信息增益
        for value in uniqueVals:    #遍历得到第i个属性的不同取值的信息熵之和
            subDataSet = splitDataSet(dataSet, i, value)    #去掉数据集中已经划分的特征取值,并返回已去除属性并符合要求的数据集
            prob = len(subDataSet)/float(len(dataSet))    #计算第i个属性为value的数据出现的次数
            newEntropy  += prob * calcShannonEnt(subDataSet)    #计算第i个属性为value的数据的信息熵
        infoGain = baseEntropy - newEntropy    #计算第i个属性的信息增益
        if (infoGain > bestInfoGain):    #判断第i个属性的信息增益是否大于0并且是最大的
            bestInfoGain = infoGain
            bestFeature = i    #记录最好属性的位置
    return bestFeature    #返回最好属性划分的位置

def majorityCnt(classList):    #递归构建决策树
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=Ture)    #将生成的分类频率字典进行降序排列
    return sortedClassCount[0][0]    #返回频率最高的一组


def createTree(dataSet,labels):    #创建树的函数代码
    classList = [example[-1] for example in dataSet]    #遍历获取标签列表
    if classList.count(classList[0]) == len(classList):    #类别完全相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:    #遍历完所有特征值时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    #选择最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]    #得到对应的标签值
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])    #清空labels[bestFeat],在下一次使用时清零
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet
                                                  (dataSet, bestFeat, value),subLabels)    #递归调用创建决策树函数
    return myTree

#inputTree  事先构建好的决策树
#featLabels  特征的名称
#testVec  待测数据的特征向量
def classify(inputTree, featLabels, testVec):    #使用决策树的分类函数
    firstStr = inputTree.keys()[0]    #得到树中的第一个特征
    secondDict = inputTree[firstStr]    #得到第一个对应的值
    featIndex = featLabels.index(firstStr)    #将标签字符串转换成索引，得到树中第一个特征对应的索引
    for key in secondDict.keys():    #遍历树
        if testVec[featIndex] == key:    #如果在secondDict[key]中找到testVec[featIndex]
            if type(secondDict[key]).__name__ == 'dict':    #判断secondDict[key]是否为字典，若为字典，递归的寻找testVec
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:    #若secondDict[key]为标签值，则将secondDict[key]赋给classLabel，返回类标签
                classLabel = secondDict[key]
    return classLabel

#决策树的序列化，可以将分类器存储在硬盘上
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)







