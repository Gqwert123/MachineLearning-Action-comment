#coding=utf-8
#选择最好的数据划分方式（程序清单3-3）

#中文注释：宫
#代码重构：宫
#环境：python2.7


from numpy import *
import operator
from os import listdir

from math import log

def createDataSet():    #简单鉴定数据集
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet, axis, value):    #划分数据集,三个参数为待划分的数据集，划分数据集的特征，特征的返回值
    retDataSet = []    #创建一个新的列表对像，防止原始数据被修改
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting #注意这里的切片用法
            reducedFeatVec.extend(featVec[axis+1:])   #注意这里两个函数的用法区别，具体查阅手册
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcShannonEnt(dataSet):    #计算给定数据集的香农熵
    numEntries = len(dataSet)    #计算数据集中的实例总数
    labelCounts = {}  #创建数据字典，其键值是最后一列的数值
    for featVec in dataSet: #the the number of unique elements and their occurance。  
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0    #如果当前键值不存在，则扩展字典并将当前键值加入字典，
        labelCounts[currentLabel] += 1                                              #每一个键值都记录了当前类别的次数
    #使用所有类标签发生的频率计算类别出现的概率，计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2 求对数
    return shannonEnt
    
def chooseBestFeatureToSplit(dataSet):    #选择最好的数据划分方式
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels  #注意列表的列表。判定当前数据集包中包含多少特征属性，最后的一个元素是列表标签
    baseEntropy = calcShannonEnt(dataSet)    #整个代码的原始香农熵，保存最初的无序度量值，用于与划分之后的数据集的熵值进行比较
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features  遍历数据集中的所有特征，使用列表推导来创建新的列表
        featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values  #set()集合去除掉列表中的重复元素
        newEntropy = 0.0
        for value in uniqueVals:     #遍历当前特征中的唯一属性值，对每一个唯一的属性值划分一次数据集，然后计算数据集的新熵值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #对所有的唯一特征值得到的熵求和   
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy  比较所有特征中的信息增益，返回特征划分最好的索引值
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

myDat, labels = createDataSet()
print(chooseBestFeatureToSplit(myDat))
print(myDat)
