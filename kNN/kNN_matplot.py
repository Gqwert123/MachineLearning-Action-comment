#coding=utf-8
import kNN
import matplotlib
import matplotlib.pyplot as plt

reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')#读取文件数据


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])#散点图使用datingDataMat矩阵的第二，第三列数据
plt.show()