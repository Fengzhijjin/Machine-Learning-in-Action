#coding:utf-8

from numpy import *

def loadDataSet():    #读取训练样本数据，并以数列的形式输出
    dataMat = []; labelMat = []    #创建空数列   dataMat为样本特征参数 labelMat为样本标签
    fr = open('testSet.txt')    #读取训练样本文件，读取数据
    for line in fr.readlines():    #循环遍历获取样本数据数列
        lineArr = line.strip().split()    #读取数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat    #返回数列

def sigmoid(inX):    #Sigmoid函数
	return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):    #实现批梯度上升算法求得参数w
    dataMatrix = mat(dataMatIn)    #将训练样本数列转换成Numpy数组
    labelMat = mat(classLabels).transpose()    #将样本标签数列进行逆置（即行向量换成列向量），并转换成Numpy数组
    m,n = shape(dataMatrix)    #读取数组的行数与列数，m为行数，n为列数
    alpha = 0.001    #alpha为梯度上升算法中的步长
    maxCycles = 500    #maxCycles为迭代次数
    weights = ones((n,1))    #返回n行1列数值为1的numpy数组
    for k in range(maxCycles):    #利用梯度上升算法求得参数w
    	h = sigmoid(dataMatrix*weights)
    	error = (labelMat - h)
    	weights = weights + alpha * dataMatrix.transpose()* error
    return weights

def plotBestFit(weights):    #通过使用Matplotlib或数决策边界及样本点分类
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)    #生成方程中，x为numpy.arange格式，并且以0.1为步长从-3.0到3.0切分
    y = (-weights[0]-weights[1]*x)/weights[2]    #拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2, x0为1,x1为x, x2为y
    ax.plot(x, y)
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):    #实现随机梯度上升算法求得参数w   迭代次数明显减少
    m,n = shape(dataMatrix)
    alpha = 0.01    #alpha为梯度上升算法中的步长
    weights = ones(n)
    for i in range(m):    #迭代了m次
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error *dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):    #实现改进的随机梯度上升算法求得参数w
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01    #alpha在每次迭代时进行调整
            randIndex = int(random.uniform(0,len(dataIndex)))    #随机选取样本数据进行更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])    #删除选取过后的样本数据
    return weights


'''算法示例：从疝气病症预测病马的死亡率'''

def classifyVector(inX, weights):    #判断函数由sigmoid函数计算值进行判断是否死亡
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():    #打开训练集、测试集，输出算法的错误率
    frTrain = open('horseColicTraining.txt')    #打开训练集
    frTest = open('horseColicTest.txt')    #打开测试集
    trainingSet = []    #特征集
    trainingLabels = []    #标签集
    for line in frTrain.readlines():    #循环遍历得到特征集与标签集
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)    #改进的随机梯度上升算法得到回归系数
    #trainWeights = gradAscent(trainingSet, trainingLabels)    #批梯度上升算法得到回归系数
    #trainWeights = stocGradAscent0(array(trainingSet), trainingLabels)    #随机梯度上升算法得到回归系数
    errorCount = 0    #预测错误计数
    numTestVec = 0.0    #测试集计数
    for line in frTest.readlines():    #循环遍历测试集数据，并判断预测是否正确并计数
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)    #计算错误率
    print ("the error rate of this test is: %f") % errorRate    #输出错误率
    return errorRate    #返回错误率

def multiTest():    #多次执行算法计算平均错误率
    numTests = 10    #进行10次预测
    errorSum=0.0    #10次预测错误率之和
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f ") % (numTests, errorSum/float(numTests))    #输出10次预测平均错误率