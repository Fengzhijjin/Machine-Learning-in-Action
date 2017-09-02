#-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir


def knn_classify(inX,dataSet,labels,k):
    '''
    knn算法
    inX 待分类向量
    dataSet 训练样本
    labels 标签向量
    k 最邻近元素的个数
    '''
    dataSetSize = dataSet.shape[0]  #numpy用法 获取训练样本数据的维度大小
    diffMat = tile(inX,(dataSetSize,1))-dataSet #将inX扩展为跟dataSet维数相同进行相减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5       #上三步用于计算欧式距离
    sortedDistIndicies = distances.argsort()  #按Distance从小到大排序 此处记录的是下标
    classCount={}
    for i in range(k):   #取前k个
        votelabel=labels[sortedDistIndicies[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1  #记录每个类别在最近的k个中出现了几个  例如 dict: {'A': 1, 'B': 2}
    sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),reverse=True)#按 value字段排列字典 
    return sortedClassCount[0][0]  #返回所属类别


def img2Vector(filename):
    '''
                将图像转换为向量  1*1024
    '''
    vector = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0,32*i+j] = int(lineStr[j])
    return vector

def getTrainSample():
    hwLabels=[]
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    classNumDic = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        classNumMax = int(fileStr.split('_')[1])
        if (classNumMax > classNumDic[classNumStr]):classNumDic[classNumStr] = classNumMax
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector('trainingDigits/%s'%fileNameStr)
    return trainingMat,hwLabels,classNumDic

def handwritingClassifyValidate(vectorUnderTest,trainingMat,hwLabels,k):
    classifierResult = knn_classify(vectorUnderTest,
                                    trainingMat,hwLabels,k)
    return classifierResult