#coding=utf-8
from numpy import * 
import kNN
import matplotlib
import matplotlib.pyplot as plt

reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')#读取文件数据


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],#散点图使用datingDataMat矩阵的第二，
15.0*array(datingLabels),15.0*array(datingLabels))#第三列数据;函数scatter()支持个性化标注散点图上面的点
plt.show()