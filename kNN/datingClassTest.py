#coding=utf-8
from numpy import * 
import kNN


reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')#读取文件数据

kNN.datingClassTest()