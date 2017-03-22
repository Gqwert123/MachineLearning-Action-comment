#coding=utf-8
from numpy import * 
import kNN


reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')#读取文件数据

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)