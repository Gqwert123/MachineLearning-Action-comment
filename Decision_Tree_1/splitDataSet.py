#coding=utf-8
#按照给定特征划分数据集（程序清单3-2）

#中文注释：宫
#代码重构：宫
#环境：python2.7


def createDataSet():    #简单鉴定数据集
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

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
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):    #划分数据集,三个参数为待划分的数据集，划分数据集的特征，特征的返回值
    retDataSet = []    #创建一个新的列表对像，防止原始数据被修改
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting #注意这里的切片用法
            reducedFeatVec.extend(featVec[axis+1:])   #注意这里两个函数的用法区别，具体查阅手册
            retDataSet.append(reducedFeatVec)
    return retDataSet


dataSet, labels = createDataSet()
print(splitDataSet(dataSet, 0, 1))