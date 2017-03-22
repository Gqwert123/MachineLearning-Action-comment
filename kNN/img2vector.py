#coding=utf-8
from numpy import * 
import kNN


reload(kNN)
testVector = kNN.img2vector('testDigits/0_13.txt')#读取文件数据
print(testVector[0,0:31])
print(testVector[0,32:63])