#coding:utf-8

from numpy import *

def loadSimpData():    #生成一个简单的数据集，标签集
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#dataMatrix  数据集
#dimen  测试数据集第dimen个特征值
#threshVal  阈值
#threshIneq  比较类别
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):    #通过阈值比较对数据进行分类
    retArray = ones((shape(dataMatrix)[0],1))    #分类记录
    #通过<和>的转换来实现对比较最大值和最小值转换
    if threshIneq == 'lt':    #比较最小值
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:    #比较最大值
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#dataArr  数据集
#classLabels  标签集
#D  权重向量
def buildStump(dataArr,classLabels,D):    #遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0    #用于在特征的所有可能值上进行遍历
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf    #初始化无穷大，后用于寻找可能的最小错误率
    for i in range(n):    #第一层for循环在数据集的所有特征上遍历
        rangeMin = dataMatrix[:,i].min()    #最小值
        rangeMax = dataMatrix[:,i].max()    #最大值
        stepSize = (rangeMax-rangeMin)/numSteps    #计算步长
        for j in range(-1,int(numSteps)+1):    #在纸上进行遍历
            for inequal in ['lt', 'gt']:    #在大于和小于之间进行转换比较
                threshVal = (rangeMin + float(j) * stepSize)    #遍历选取阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)    #数据比较结果
                errArr = mat(ones((m,1)))    #判断错误记录，判断错误为1
                errArr[predictedVals == labelMat] = 0    #判断正确为0
                weightedError = D.T*errArr    #计算加权错误率
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                # (i, threshVal, inequal, weightedError))
                if weightedError < minError:    #当前错误率与记录错误率比较，记录较小错误率时数据
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#dataArr  数据集
#classLabels  标签集
#numIt  迭代次数
def adaBoostTrainDS(dataArr,classLabels,numIt=40):    #
    weakClassArr = []    #单层决策树数组
    m = shape(dataArr)[0]    #数据集中数据个数
    D = mat(ones((m,1))/m)    #每个数据的权重，初始值相同
    aggClassEst = mat(zeros((m,1)))    #记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)    #得到最小错误率的单层决策树，同时返回错误率和类别向量
        #print ("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))    #本次单层决策树输出结果的权重
        bestStump['alpha'] = alpha    #将值添加到bestStump字典中
        weakClassArr.append(bestStump)    #将字典添加到weakClassArr列表中
        #print ("classEst:",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)    #以下三行用于计算下一次迭代中的新权重向量D
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst    #以下四行为错误率累加计算
        #print ("aggClassEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error:",errorRate)
        if errorRate == 0.0:    #如果错误率为0则结束for循环
            break
    return weakClassArr,aggClassEst

#datToClass  待分类样例
#classifierArr  弱分类器组成的数组
def adaClassify(datToClass,classifierArr):    #利用多个弱分类器进行分类
    dataMatrix = mat(datToClass)    #数据集
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))    #记录每个数据点的类别估计累计值
    for i in range(len(classifierArr)):    #遍历classifierArr中的所有弱分类器
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #计算第i个弱分类器的分类结果
        aggClassEst += classifierArr[i]['alpha']*classEst    #计算第i个弱分类器在预测结果中的权重
        #print (aggClassEst)    #打印弱分类器的权重
    return sign(aggClassEst)    #返回预测结果类别

#在马病数据集上应用AdaBoost分类器

def loadDataSet(fileName):    #自适应数据加载函数，该函数可自动检测特征数目，同时假定最后一个特征是类别标签
    numFeat = len(open(fileName).readline().split('\t'))    #计算列数
    dataMat = []    #数据集
    labelMat = []    #标签集
    fr = open(fileName)
    for line in fr.readlines():    #遍历每一行
        lineArr = []
        curLine = line.strip().split('\t')    #将每一行中的数据组成一个列表
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)    #特征数据添加到数据集中
        labelMat.append(float(curLine[-1]))    #类别数据添加到标签集中
    return dataMat,labelMat


#绘制ROC曲线及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is:",ySum*xStep)