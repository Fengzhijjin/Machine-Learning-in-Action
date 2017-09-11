#coding:utf-8

from numpy import *

def loadDataSet(fileName):    #读取文件数据函数，返回文件中的数据列表
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')    #文件中每行中的数据是以“tab”键分割的浮点数
        fltLine = map(float, curLine)
        dataMat.append(fltLine)    #添加到数据列表中
    return dataMat

def distEclud(vecA, vecB):    #距离计算函数，计算两个向量的欧式距离
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):    #在给定数据集中构建一个包含k个随机质心的集合
    n = shape(dataSet)[1]    #数据集中有n个特征
    centroids = mat(zeros((k, n)))    #创建k个质心
    for j in range(n):    #遍历求得每种特征的取值范围，并在范围内选取k个数据为质点特征值
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

#dataSet  数据集
#k  簇数目
#distMeas  距离计算函数
#createCent  创建初始质心函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):    #K-均值算法函数
    m = shape(dataSet)[0]    #数据集中数据点个数为m
    clusterAssment = mat(zeros((m, 2)))    #簇分配结果矩阵，第一列簇索引值，第二列存储误差（指该点到簇质心距离）
    centroids = createCent(dataSet, k)    #初始化k个质点
    clusterChanged = True
    while clusterChanged:    #当clusterChanged为True是继续循环，为False（即任意点的簇分配结果未发生变化）时终止循环
        clusterChanged = False
        for i in range(m):    #循环遍历数据集中的每个数据点，计算各质点到它的距离，重新分配簇
            minDist = inf
            minIndex = -1
            for j in range(k):    #寻找距离数据点最近的质点
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:    #判断数据点分配簇是否更改，发生更改后修改clusterAssment=True
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print centroids
        for cent in range(k):    #更新质点的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment    #返回质点和簇分配矩阵

#二分K-均值聚类算法
#dataSet  数据集
#k  簇数目
#distMeas  距离计算函数
def biKmeans(dataSet, k, distMeas=distEclud):    #二分K-均值聚类算法函数
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))    #创建一个矩阵存储数据集中每个点的分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]    #创建一个簇寻找质心，并存储（mean为求均值函数zxis=0则为求列均值）
    centList = [centroid0]
    for j in range(m):    #存储所有数据点到质点的误差
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf    #最小SSE值
        for i in range(len(centList)):    #遍历簇列表中的每一个簇，
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]    #寻找数据集中分配簇为i所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)    #调用k-均值算法进行二分类
            sseSplit = sum(splitClustAss[:,1])    #计算二分配的SSE值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])    #计算分配簇中非i簇的SSE值
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:    #判断两个SSE值之和是否小于小于最小SSE值
                bestCentToSplit = i    #选取更改第i个质点
                bestNewCents = centroidMat    #第i个质点的特征集
                bestClustAss = splitClustAss.copy()    #更新后第i个簇变为两个新簇
                lowestSSE = sseSplit + sseNotSplit    #更新最小SSE值
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)    #在生成的二分簇中，将第二个簇号改为簇数加一
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit    #在生成的二分簇中，将第一个簇号改为i
        print ('the bestCentToSplit is: ', bestCentToSplit)    #输出要做二分操作的簇号
        print ('the len of bestClustAss is: ', len(bestClustAss))    #输出属于要做二分操作的簇的数据点个数
        centList[bestCentToSplit] = bestNewCents[0,:]    #将二分操作中第一个质点更新为第i个质点数据
        centList.append(bestNewCents[1,:])    #添加二分操作中第二个质点的数据
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss    #更新簇分配矩阵中的数据
    return centList, clusterAssment