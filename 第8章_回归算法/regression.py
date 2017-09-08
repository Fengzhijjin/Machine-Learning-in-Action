#coding:utf-8

from numpy import *

def loadDataSet(fileName):    #该函数打开一个用tab键分隔的文本文件，默认文件的每行的最后一个值是目标值
    numFeat = len(open(fileName).readline().split('\t')) - 1    #计算文件中的特征数
    dataMat = []    #数据集
    labelMat = []    #标签集
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')    #将文件中的每行生成列表
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):    #计算最佳拟合曲线
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:    #判断行列式是否为0
        print("This matrix is singular , cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)    #求解w的最优解
    return ws

#xMat  数据集
#yMat  标签集
#ws标准回归算法计算出的w向量
def plotRegress00(xMat,yMat,ws):    #绘制标准回归拟合曲线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

#xArr  数据集
#yArr 标签集
#yHat 局部加权线性回归算法的预测值
def plotRegress01(xArr,yArr,yHat):    #绘制局部加权线性回归拟合曲线
    import matplotlib.pyplot as plt
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

#通过核函数改进得到的局部加权线性回归函数

def lwlr(testPoint,xArr,yArr,k=1.0):    #利用核函数改进的局部加权线性回归函数求得一点的预测值
    xMat = mat(xArr)    #数据集
    yMat = mat(yArr).T    #标签集
    m = shape(xMat)[0]
    weights = mat(eye((m)))    #创建权重矩阵
    for j in range(m):    #计算权重，权重值大小以指数级衰减
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:    #判断行列式是否为0
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))    #求解回归系数
    return testPoint * ws    #返回预测值

def lwlrTest(testArr,xArr,yArr,k=1.0):    #循环遍历使用lwlr函数求得整个数据集的预测值
    m = shape(testArr)[0]
    yHat = zeros(m)    #创建预测值数列
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):    #计算预测值与真实值的误差（误差以实际误差平方表示）
    return ((yArr-yHatArr)**2).sum()

#岭回归求解回归函数
#岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计

def ridgeRegres(xMat,yMat,lam=0.2):    #给定lambda下的岭回归求解，用于计算回归系数
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)    #回归系数计算
    return ws

def ridgeTest(xArr,yArr):    #在30个不同的lambda下调用ridgeRegres函数求解30组回归系数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    #以下四行为数据标准化
    yMat = yMat-yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

#前向逐步线性回归算法

# 按照均值为0，方差为1，对列进行标准化处理
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    # 求方差
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat

#xArr  输入数据
#yArr  预测变量
#eps  每次迭代调整的步长
#numIt  迭代次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))    #保存W的值
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#实例：预测乐高玩具套装的价格

from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?\
    key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr,setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv :
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc,sellingPrice))
                    retX.append([yr,numPce,newFlag,origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)

def setDataCollect(retX,retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

