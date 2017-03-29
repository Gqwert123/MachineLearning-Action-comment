#coding=utf-8
#创建树的函数代码（程序清单3-4）

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

def majorityCnt(classList):    #投票表决(如果数据集已经处理了所有的属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点)
    classCount={}      
    for vote in classList:   #使用分类名称的列表
        if vote not in classCount.keys(): classCount[vote] = 0    #创建键值为classList中唯一值的数据字典
        classCount[vote] += 1    #字典对象存储了classList中每个类标签出现的频率
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)    #利用operator操作键值排序字典  iteritem（）返回访问元素的访问迭代器
    return sortedClassCount[0][0]    #返回出现次数最多的分类名称

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


def splitDataSet(dataSet, axis, value):    #划分数据集,三个参数为待划分的数据集，划分数据集的特征，特征的返回值
    retDataSet = []    #创建一个新的列表对像，防止原始数据被修改
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting #注意这里的切片用法
            reducedFeatVec.extend(featVec[axis+1:])   #注意这里两个函数的用法区别，具体查阅手册
            retDataSet.append(reducedFeatVec)
    return retDataSet

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]    #包含数据集的所有类标签
    if classList.count(classList[0]) == len(classList):     #所有类标签完全相同，递归停止
        return classList[0]
    if len(dataSet[0]) == 1:   #使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，递归停止
        return majorityCnt(classList)    #返回次数出现最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #创建树
    myTree = {bestFeatLabel:{}}  #字典存储树的信息
    del(labels[bestFeat])     #每次划分一个树节点后，都要删除一个之前划分节点的属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #遍历当前选择特征包含的所有的属性值，在每个数据集上递归调用函数createTree（），得到的返回值将被插入到字典变量myTree中
    for value in uniqueVals:
        subLabels = labels[:]       #为了保证每次调用函数creatTree（）不改变原始列表内容，使用新的变量subLabels代替原始列表
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree  


myDat, labels = createDataSet()
myTree = createTree(myDat,labels)
print(myTree)