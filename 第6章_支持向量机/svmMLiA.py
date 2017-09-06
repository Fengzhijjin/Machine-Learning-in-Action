#coding:utf-8

from numpy import *

def loadDataSet(fileName):    #读取训练样本数据，并以数列的形式输出
    dataMat = []    #特征集
    labelMat = []    #标签集
    fr = open(fileName)    #打开训练样本集文件
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):    #随机选取选取一个不等于i的alpha下标
    j=i
    while (j==i):
        j = int(random.uniform(0,m))    #随机选取0-（m-1)的数字
    return j

def clipAlpha(aj,H,L):    #用于调整大于H或小于L的alpha值
    if aj > H:    #大于H的值改为H
        aj = H
    if L > aj:    #小于L的值改为L
        aj = L
    return aj

#dataMatIn: 特征集
#classLabels: 标签集
#C: 常数C
#toler: 容错率
#maxIter: 退出前最大的循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):    #计算b和alphas值
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))    #创建一个alpha列矩阵
    iter = 0    #循环次数记录
    while (iter < maxIter):
        alphaPairsChanged = 0     #每次循环初始化为0,用于计算alpha是否已经优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b    #fXi为预测的类别
            Ei = fXi - float(labelMat[i])    #预测值与真实值误差，计算误差值Ei
            #如果误差很大，那么可对该数据实例所对应的alpha值进行优化，分别对正间隔和负间隔做了测试并且检查了zlpha值，
            #保证其不能等于0或C，，由于后面的alpha小于0或者大于C时将被调整为0或C，所以一旦该if语句中他们等于这两个值的话，
            #那么他们就已经在“边界”上了，因而不能够减小或增大，因此就不值得对他们进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #误差较大进行优化
                j = selectJrand(i,m)    #随机选取第二个alpha值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b    #fXj为第二个值的预测类别
                Ej = fXj - float(labelMat[j])    #预测值与真实值误差
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #L和H用于将alphas[j]调整到0-C之间。如果L==H，就不会做任何改变，直接执行continue
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[j])
                if L==H:
                    print ("L==H")
                    continue
                #eta是alphas[j]的最优修改量，如果eta==0,需要退出for循环的当前迭代过程
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]\
                                                                                                    *dataMatrix[j,:].T
                if eta >= 0:
                    print ("eta>=0")
                    continue
                #计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j]*(Ei -Ej)/eta
                #并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j],H,L)
                #检查alphas[j]是否有轻微改变，如果是的话，退出for循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not movingenough")
                    continue
                #然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #在对alphas[i]和alphas[j]进行优化后，给这两个alpha值设置一个常数项b
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*\
                                                                (alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*\
                                                                (alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d") % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print ("iteration number: %d") % iter
    return b,alphas

'''利用完整Platt SMO算法加速优化'''

'''
class optStruct:    #创建一个类，用于创建对象存储数据
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn    # dataMatIn: 特征集
        self.labelMat = classLabels    # classLabels: 标签集
        self.C = C    # C: 常数C
        self.tol = toler    # toler: 容错率
        self.m = shape(dataMatIn)[0]    # maxIter: 退出前最大的循环次数
        self.alphas = mat(zeros((self.m,1)))    #alphas值
        self.b = 0    #b值
        self.eCache = mat(zeros((self.m,2)))    #误差缓存 第一列是eCache是否有效的标志位，第二列是实际的误差E值
'''

class optStruct:    # 使用Kernel函数的class optStruct,用于创建对象存储数据
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn    # dataMatIn: 特征集
        self.labelMat = classLabels    # classLabels: 标签集
        self.C = C    # C: 常数C
        self.tol = toler    # toler: 容错率
        self.m = shape(dataMatIn)[0]    # maxIter: 退出前最大的循环次数
        self.alphas = mat(zeros((self.m,1)))    #alphas值
        self.b = 0    #b值
        self.eCache = mat(zeros((self.m,2)))    #误差缓存 第一列是eCache是否有效的标志位，第二列是实际的误差E值
        self.K = mat(zeros((self.m,self.m)))    #kTup是一个包含核函数信息的元组
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# kTup: 元组，核函数的信息，元组的一个参数是描述所用核函数类型的一个字符串，另外2个参数则都是核函数可能需要的可选参数
# 一个调用实例：kernelTrans(sVs, dataMat[i,:], ('rbf', k1)),其中k1是径向基核函数高斯版本中的sigma
def kernelTrans(X, A, kTup) :    # 核转换函数
    m,n = shape(X)
    K = mat(zeros((m, 1)))    # 构建一个列向量
    if kTup[0] == 'lin' :    # 检查元组以确定核函数的类型
        K = X * A.T
    elif kTup[0] == 'rbf' :     # 在径向基核函数的情况下
        for j in range(m) :    # for循环中对于矩阵的每个元素计算高斯函数的值
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))    # 将计算过程应用到整个向量，元素间的除法
    else :
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

'''
def calcEk(oS, k):    #计算给定alpha值的误差E值
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b    #预测类型
    Ek = fXk - float(oS.labelMat[k])    #误差值
    return Ek
'''

def calcEk(oS, k):    # 使用Kernel函数的calcEk,计算给定alpha值的误差E值
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)    #预测类型
    Ek = fXk - float(oS.labelMat[k])    #误差值
    return Ek

def selectJ(i, oS, Ei):    #用于选择第二个alpha值并返回其对应的误差E值
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]    #将输入值Ei在缓存中设置成为有效的
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]    #nonzero()返回一个列表，列表包含以输入列表为目录的列表值，这里的值并非\
                                                                              # 零返回的非零E值对应的alpha值，而不是E值本身
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):    # 选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:    # 第一次循环，随机选择一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej    #返回第二个alpha值并计算误差

def updateEk(oS, k):    #计算误差值并存入对象内存中
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

'''
def innerL(i, oS):    #选择第一个alpha值的外循环
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol)\
                                                                    and (oS.alphas[i] > 0)):    #误差较大时进行优化
        j,Ej = selectJ(i, oS, Ei)    #选取第二个alpha值并计算误差
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print ("L==H")
            return 0
        # eta是alphas[j]的最优修改量
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print ("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)    #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print ("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)    #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]\
                                                                        *(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]\
                                                                        *(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0
'''

def innerL(i, oS) :    # 使用Kernel函数的innerL,选择第一个alpha值的外循环
    Ei = calcEk(oS, i)
    if ( (oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C) ) or ( (oS.labelMat[i]*Ei > oS.tol)\
                                                                         and (oS.alphas[i] > 0) ) :    #误差较大时进行优化
        j, Ej = selectJ(i, oS, Ei)    #选取第二个alpha值并计算误差
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
        if oS.labelMat[i] != oS.labelMat[j] :
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else :
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H :
            print("L==H")
            return 0
        eta = 2.0*oS.K[i,j] - oS.K[i,i] - oS.K[j,j]    # eta是alphas[j]的最优修改量
        if eta >=0 :
            print ("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)    #更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001 :
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)    #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]\
                                                                                               - alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]\
                                                                                               - alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]) :
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]) :
            oS.b = b2
        else :
            oS.b = (b1 + b2) / 2.0
        return 1
    else :
        return 0

#完整版Platt SMO的外循环代码
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 退出循环条件：1、迭代次数超过指定最大值；2、遍历整个集合都未对任意alpha对进行修改。
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            # 在数据集上遍历任意可能的alpha，使用innerL()来选择第二个alpha，并在可能时对其进行优化
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print ("fullSet, iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历所有非边界alpha值，也就是不在边界0或C上的值
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calacWs(alphas,dataArr,classLabels):    # 利用alpha值，进行分类
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 利用核函数进行分类的径向基测试函数,此函数从文件中读取数据集，然后在该数据集上运行Platt SMO算法，其中核函数的类型是'rbf'
def testRbf(k1 = 1.3) :    # k1: 高斯径向基函数中一个用户定义变量
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m) :    #for循环中前两行，给出了如何利用核函数进行分类
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]) :
            errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m) :
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]) :
            errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))

# 基于SVM的手写数字识别

def img2vector(filename):   #将手写图片转换为向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

# 利用核函数进行分类的径向基测试函数,此函数从文件中读取数据集，然后在该数据集上运行Platt SMO算法，并调用loadImages（）函数\
# 来获取类别标签和数据
def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print ("the test error rate is : %f" % (float(errorCount)/m))