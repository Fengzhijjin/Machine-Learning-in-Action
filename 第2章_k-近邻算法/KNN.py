#coding:utf-8
'''
KNN算法实现及实例应用
python 2.7  代码实现
'''

from numpy import *
import operator
from os import listdir

def createDataSet():    #生成样本数据集和标签函数
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#inx  数据向量
#dataSet  训练样本集
#labels  标签向量
#k  用于选择最近邻居的数量
def classify0(inX, dataSet, labels, k) :    #k-近邻算法输出对输入数据的判断
    dataSetSize = dataSet.shape[0]
    '''shape()函数：numpy库中的方法，用于计算一个多维数组的维度，shape(0)返回第一维度数，shape(1)返回第二维度数 以此类推 '''
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    ''' 此句的意思是  将inX二维数组化，inX是数组化的参数，(dataSetSize,1)  dataSetSize是数组化后的行数，1是数组化后的列数
        tile()的使用例子
        >>> c = np.array([1,2,3,4])
        >>> np.tile(c,(4,1))
        array([[1, 2, 3, 4],
               [1, 2, 3, 4],
               [1, 2, 3, 4],
               [1, 2, 3, 4]])
        >>> a = np.array([0, 1, 2])
        >>> np.tile(a, 2)
        array([0, 1, 2, 0, 1, 2])
        >>> np.tile(a, (2, 2))
        array([[0, 1, 2, 0, 1, 2],
               [0, 1, 2, 0, 1, 2]])
    '''
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    '''sqDiffMat.sum(axis=1)  是数组求和  sxis=1时 表示数组行之间数求和
       axis=0时  表示数组列之间数求和
    '''
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    '''distances.argsort()  表示对distances数组进行非降序排序（即升序排序）返回的是升序排列后数组元素的下标
    例如：
        >>> x = np.array([3, 1, 2])
        >>> np.argsort(x)
        array([1, 2, 0])
    '''
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    '''for循环语句的意思是排出数组中各个标签的数量'''
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    '''将生成的分类频率字典进行降序排列，reverse=False表示为升序排序，reverse=True表示为降序排序
        classCount.iteritems()决定了排序的参数是classCount字典中的键值对（即键与值）
        key=operator.itemgetter(1)表示的是以排序参数中的第二个元素为关键字排序
        reverse=True 表示进行降序排列，reverse=False 表示进行升序排列
    '''
    return sortedClassCount[0][0]

#实例：使用k-近邻算法改进约会网站的配对效果

def file2matrix(filename):    #文件解析函数（输入文件输出训练样本矩阵和类标签向量）
    fr = open(filename)    #打开文件
    arrayOLines = fr.readlines()    #读取文件所有行
    numberOfLines = len(arrayOLines)    #计算数据点数量
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:    #遍历文件各行
        line = line.strip()
        listFromLine = line.split('\t')    #将每行数据以空格进行分割
        returnMat[index,:] = listFromLine[0:3]    #将文件每行特征值存储进训练样本矩阵
        classLabelVector.append(int(listFromLine[-1]))    #将文件每行标签值存入类标签向量
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):    #归一化特征值函数
    minVals = dataSet.min(0)    #求取每列最小值并存储
    maxVals = dataSet.max(0)    #求取每列最大值并存储
    ranges = maxVals - minVals    #求取没列的取值范围
    normDataSet = zeros(shape(dataSet))    #创建归一化特征值列表
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))    #特征值减去最小值
    normDataSet = normDataSet/tile(ranges,(m,1))    #除以取值范围
    return normDataSet,ranges,minVals    #返回归一化特征值，取值范围，最小值

def datingClassTest():    #分类器针对约会网站的测试函数
    hoRatio = 0.10    #用于测试的数据所占比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')    #读取数据
    normMat, ranges, minVals = autoNorm(datingDataMat)    #归一化数据的特征值
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)    #计算测试数据数量
    errorCount = 0.0    #错误数标记
    for i in range(numTestVecs):    #遍历测试数据
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)    #利用KNN函数求得预测值
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))    #打印预测值与标签值
        if (classifierResult != datingLabels[i]):    #如果预测值不等于标签值进行记录
            errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))    #输出错误率

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year:"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person:", resultList[classifierResult - 1])

#实例：手写系统识别

def img2vector(filename):   #将手写图片转换为向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():    #手写数字识别系统
    hwLabels = []    #初始化类别标签为空列表
    trainingFileList = listdir('trainingDigits')    #列出给定目录下所有训练数据的文件名
    m = len(trainingFileList)     #求训练数据数目
    trainingMat = zeros((m,1024))     #初始化m个图像的训练矩阵
    for i in range(m):    #遍历每一个训练数据
        fileNameStr = trainingFileList[i]    #取出一个训练数据的文件名
        fileStr = fileNameStr.split('.')[0]      #去掉该训练数据的后缀名.txt
        classNumStr = int(fileStr.split('_')[0])     #取出代表该训练数据类别的数字
        hwLabels.append(classNumStr)     #将代表该训练数据类别的数字存入类别标签列表
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)    #调用图像转换函数将该训练数据的输入特征转换为向量\
        # 并存储
    testFileList = listdir('testDigits')    #列出给定目录下所有测试数据的文件名
    errorCount = 0.0    #初始化测试犯错的样本个数
    mTest = len(testFileList)    #求测试数据数目
    for i in range(mTest):    #遍历每一个测试数据
        fileNameStr = testFileList[i]    #取出一个测试数据的文件名
        fileStr = fileNameStr.split('.')[0]    #去掉该测试数据的后缀名.txt
        classNumStr = int(fileStr.split('_')[0])    #取出代表该测试数据类别的数字
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)    #调用图像转换函数将该测试数据的输入特征转换为向量
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)    #调用k-NN简单实现函数，并返回分类器对该\
        # 测试数据的分类结果
        print ("the classifier came back with: %d,the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount +=1.0    #统计分类器对测试数据分类犯错的个数
    print ("\nthe total numberof errors is: %d" % errorCount)    #输出分类器错误数
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))    #输出分类器错误率












