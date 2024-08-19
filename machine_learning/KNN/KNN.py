# 导入包
import matplotlib.font_manager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import operator
import os

Ch_font = matplotlib.font_manager.FontProperties(fname = "E:/My_resources/Fonts/source-han-serif-1.001R/OTF/SimplifiedChinese/SourceHanSerifSC-Bold.otf")


# 生成测试样本
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# kNN实现一个简单的判断器
def classify0(inX, dataSet, labels, k):
    """inX 为输入的待判断数据"""
    # 计算距离（每个特征值都要）
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # 每个特征值距离累加
    distances = sqDistances**0.5 # 等于开根号

    # 距离排序
    sortedDistIndicies = distances.argsort()     

    # 计算前k近的出现最多频率的类别
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 构建将文本数据转化为分类器可解释的格式（matrix）
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file，获取文本文件的行数
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return，初始化返回的矩阵格式，行和特征值数量
    classLabelVector = []                       #prepare labels return   ，初始化返回标签
    fr = open(filename) # 重新打开文件
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t') # 以制表符分割
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) # 每行最后一个数据是标签
        index += 1
    return returnMat,classLabelVector

# 特征值归一化
def autoNorm(dataSet):
    minVals = dataSet.min(axis = 0) # 取列最小值
    maxVals = dataSet.max(axis = 0) # 取列最大值
    ranges = maxVals - minVals # 生成范围
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1)) # 现在其实numpy可以广播
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

# 计算错误率
def datingClassTest():
    hoRatio = 0.50      #hold out 50%，测试50%的数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data set from file，加载数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# 构建输入自己的数据，匹配最佳约会对象的好感度
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals =autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals) / ranges,normMat,datingLabels,3)
    print('you will probably like this person:',resultList[classifierResult - 1])

# 构建图像转化矩阵器
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 构建训练函数
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')           #load the training set，加载训练集，形成分类器
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    
    testFileList = os.listdir('digits/testDigits')        #iterate through the test set，获取测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))