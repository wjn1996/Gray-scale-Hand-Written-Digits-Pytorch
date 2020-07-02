import numpy as np
#使用listdir模块，用于访问本地文件
from os import listdir
from sklearn import neighbors
import sklearn.tree as dtree
from sklearn import svm

class KNN():
    # K近邻算法
    # acc:0.9705
    def __init__(self, train_x, train_y, test_x, test_y):
        self.algorithm = 'kd-tree'
        self.n_neighbors = 3
        self.name = 'KNN (' + self.algorithm + ', n_neighbors:' + str(self.n_neighbors) + ')'
        self.train_x = self.process(np.array(train_x))
        self.train_y = self.process(np.array(train_y))
        self.test_x = self.process(np.array(test_x))
        self.test_y = self.process(np.array(test_y))
        self.knn = neighbors.KNeighborsClassifier(algorithm=self.algorithm,n_neighbors=self.n_neighbors)

    def process(self, x):
        return x.reshape(len(x), -1)

    def train(self):
        self.knn.fit(self.train_x, self.train_y)

    def test(self):
        self.res = self.knn.predict(self.test_x)
        acc = self.knn.score(self.test_x, self.test_y)
        return acc

class DT():
    # 决策树算法
    # acc(depth = 16) 0.7596
    def __init__(self, train_x, train_y, test_x, test_y):
        self.max_depth = 16
        self.name = 'Dicision Tree (depth:' + str(self.max_depth) + ')'
        self.train_x = self.process(np.array(train_x))
        self.train_y = self.process(np.array(train_y))
        self.test_x = self.process(np.array(test_x))
        self.test_y = self.process(np.array(test_y))
        self.dtree = dtree.DecisionTreeRegressor(max_depth=self.max_depth)

    def process(self, x):
        return x.reshape(len(x), -1)

    def train(self):
        self.dtree.fit(self.train_x, self.train_y)

    def test(self):
        self.res = self.dtree.predict(self.test_x)
        acc = self.dtree.score(self.test_x, self.test_y)
        return acc


class SVM():
    # 决策树算法
    # acc 0.9792
    def __init__(self, train_x, train_y, test_x, test_y):
        
        self.name = 'SVM'
        self.train_x = self.process(np.array(train_x))
        self.train_y = self.process(np.array(train_y))
        self.test_x = self.process(np.array(test_x))
        self.test_y = self.process(np.array(test_y))
        self.svm = svm.SVC()

    def process(self, x):
        return x.reshape(len(x), -1)

    def train(self):
        self.svm.fit(self.train_x, self.train_y)

    def test(self):
        self.res = self.svm.predict(self.test_x)
        acc = self.svm.score(self.test_x, self.test_y)
        return acc

#构建KNN分类器：设置查找算法以及邻居点 数量(k)值。
#KNN是一种懒惰学习法，没有学习过程，只在预测时去查找最近邻的点，
#数据集的输入就是构建KNN分类器的过程



# #测试集评价
# dataSet,hwlLabels =readDataSet('testDigits')
# res=knn.predict(dataSet) #对测试集进行预测
# error_num =np.sum(res!=hwlLabels)   #统计预测错误的数目
# num =len(dataSet) #测试集的数目

# print("Total num:",num,"Wrong num:",error_num," WrongRate:",error_num/float(num))