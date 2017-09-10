#coding:utf-8

from numpy import *

class treeNode():    #树节点类
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat    #待切分的特征名
        valueOfSplit = val    #待切分的特征值
        rightBranch = right    #右子树
        leftBranch = left    #左子树

def loadDataSet(fileName):    #读取数据文件
    dataMat = []
    fr = open(fileName)    #打开文件
    for line in fr.readlines():    #循环读取文件中的每一行
        curLine = line.strip().split('\t')    #将一行中的数据以tab键为分割符，进行分割
        fltLine = map(float,curLine)    #将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

#dataSet  数据集合
#feature  待切分的特征
#value  该特征的某个值
def binSplitDataSet(dataSet, feature, value):    #通过在给定特征和特征值的情况下，将数据集合切分成两个子集
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):    #生成叶节点模型  （在回归树中，该模型其实就是目标变量的均值）
    return mean(dataSet[:,-1])

def regErr(dataSet):    #误差估计函数 返回目标变量的总方差
    return var(dataSet[:,-1]) * shape(dataSet)[0]    #var函数可直接求均方差

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):    #回归树构建的核心函数，找到数据的最佳二元切分方法
    tolS = ops[0]    #tolS  容许的误差下降值
    tolN = ops[1]    #tolN  切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:    #统计所剩特征值的种类，只剩一种时无需切分直接返回
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:    #如果切分数据集后误差减小不大，则直接创建叶节点
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):    #如果切分数据集后，某个子集的大小小于用户定义的参数tolN后\
                                                                                                  # 则直接创建叶节点
        return None, leafType(dataSet)
    return bestIndex,bestValue    #返回切分特征和特征值

#dataSet  数据集
#leafType  建立叶节点的函数
#errType  误差计算函数
#ops  一个包含树构建所需其它参数的数组
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):    #树构建函数
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)    #切分函数判断数据集是否能分成两部分
    if feat == None:    #如果是满足不能再切分条件，则返回叶节点值
        return val
    #如果能切分成两部分则，用字典表示两部分并递归调用createTree（）函数继续构建
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)    #使用binSplitDataSet（）函数切分成两个子集
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

#回归树减枝函数
def isTree(obj):    #测试输入变量是否是一棵树（而不是叶节点）返回布尔类型结果
    return (type(obj).__name__=='dict')

def getMean(tree):    #此函数从上到下遍历树直到叶节点，如果找到两个叶节点则计算他们的平均值，该函数对数进行塌陷处理（则返回树平均值）
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

#tree  待减枝的树
#testData  减枝所需的测试数据
def prune(tree, testData):
    if shape(testData)[0] == 0:    #判断测试集是否为空（没有测试数据则对树进行塌陷处理
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):    #判断左右子树是否任意一个为树
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):    #判断左子树是否为树，当左子树为树时递归调用prune（）函数
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):    #判断右子树是否为树，当右子树为树时递归调用prune（）函数
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):    #判断左右子树是否都不位树（即都为叶节点）
        #以下四行用于计算合并子树后误差值与当前不合并的误差值
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:    #判断合并误差值是否小于不合并的误差值，小于时合并返回左右叶节点平均值
            print "merging"
            return treeMean
        else:    #不小于时返回左右节点树
            return tree
    else:
        return tree

#模型树构建

def linearSolve(dataSet):    #将数据集格式化成目标变量Y和自变量X，返回回归方程w值ws，自变量X，因变量Y
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:    #判断行列式是否为0
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)    #公式计算w
    return ws,X,Y

def modelLeaf(dataSet):    #当函数不需要切分时，生成叶节点模型
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):    #在给定的数据集上计算误差
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

#示例：树回归与标准回归的比较

#用树回归进行预测

def regTreeEval(model, inDat):    #对回归树叶节点进行预测时，输入单个数据点或行向量，返回浮点数
    return float(model)

def modelTreeEval(model, inDat):    #对模型树叶节点进行预测时，对输入数据进行格式化处理，计算预测值
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):    #自顶向下遍历整棵树，直到命中叶节点，返回预测值
    if not isTree(tree):
        return modelEval(tree, inDat)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):    #调用treeForeCast函数以向量的形式返回一组预测值
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
















