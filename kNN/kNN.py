#coding=utf-8
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir #函数listdir可以列出给定目录的文件名


#中文注释：宫
#环境：python2
#sublime Text3运行正常


#k近邻算法
def classify0(inX, dataSet, labels, k):#输入向量，输入的训练样本集，标签向量，选择最近邻的数目
    #距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


#处理输入格式问题，输入为文件名字符串，输出为训练样本矩阵和类标签向量
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file得到文本行数
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return创建以零填充的矩阵，为了简化，另外的一个维度设为3
    classLabelVector = []                       #prepare labels return   
    #解析文件数据到列表，循环处理文件中的每一行的数据
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()#截取掉所有的回车字符
        listFromLine = line.split('\t')#将整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]#选取前三个元素，存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#归一化特征值    
def autoNorm(dataSet):
    minVals = dataSet.min(0)#每一列的最小值
    maxVals = dataSet.max(0)#每一列的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]#数组的大小  
    normDataSet = dataSet - tile(minVals, (m,1))#注意事项：特征值矩阵有1000*3个值。而minVals和range的值都为1*3.为了解决这个问题使用numpy中tile函数将变量内容复制成输入矩阵同样大小的矩阵  
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide矩阵除法
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50      #hold out 10% ?
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #计算测试向量的数量，此步决定哪些数据用于分类器的测试和训练样本
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    #计算错误率，并输出结果
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount


#把一个32x32的二进制图像矩阵转换为1x1024的向量  
def img2vector(filename):
    returnVect = zeros((1,1024))#创建1x1024的numpy数组
    fr = open(filename)
    for i in range(32):#循环读取前32行，并且将每行头32个字符值存储在numpy数组中，最后返回数组
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)   #目录中有多少文件
    trainingMat = zeros((m,1024))    #创建一个m行1024列的训练矩阵，该矩阵的每一行存储一个图像
    #从文件名解析出分类数字，该目录的文件按照规则命名，如文件9_45.txt的分类是9，它是数字9的第45个实例
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)    #类代码存储到变量hwLabels中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    #对testDigits执行相似操作，但是不载入矩阵，而是使用classify（）函数测试该目录下的每一个文件
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
