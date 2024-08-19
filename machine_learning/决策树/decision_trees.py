# 导入包
import numpy as np
import matplotlib.pyplot as plt
import os
import operator
import math

# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance，为所有可能的分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算熵    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2) #log base 2
    return shannonEnt


# 生成简单数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting，选出符合特征的数据1，取得该特征存进新的列表里
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):

    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels，获取有多少个特征（实际上最后一个元素是类别标签）
    baseEntropy = calcShannonEnt(dataSet) # 初始熵
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):        #iterate over all the features，遍历所有特征（最后元素不包含）
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values，构建当前特征有多少个特征值(即多少种可能值)

        newEntropy = 0.0 # 初始化新的熵
        # 计算当前特征不同特征值划分的熵，并累加为此特征划分的熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) # 累加
         
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy，计算最好的信息增益，即让熵减少最大
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

# 多数表决决定类型不一致的叶子节点的分类
def majorityCnt(classList):
    """classList为输入的label列表"""
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 递归树的构建
def createTree(dataSet,labels):
    """注意涉及列表的删除操作，所以第一次不要传递真实值(或者到时候重新获取labels即可)"""
    # 获取每个实例的所属标签
    classList = [example[-1] for example in dataSet]

    # 递归终止条件一：所有实例属于相同的类别
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    
    # 递归终止条件二：仅有一个特征了
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList) # 返回多数表决的结果
    
    # 继续划分数据集
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 以某个特征划分后，删除对应特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) # 记录特征值的个数
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                   

# 使用已经建立好的决策树
def classify(inputTree,featLabels,testVec):
    # testVec为单个测试向量
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr] # 获取子节点数据
    featIndex = featLabels.index(firstStr) # 将标签字符串转化为索引
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel