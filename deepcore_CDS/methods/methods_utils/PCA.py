
# coding:utf-8
"""
Author: zhaoxingfeng	Data: 2017.02.28    Source link:https://github.com/zhaoxingfeng/PCA/blob/master/PCA.py
Function: Principal Component Analysis(PCA)
Version: V2.0
Reference：
[1]进击的马斯特.浅谈协方差矩阵[DB/OL].http://pinkyjie.com/2010/08/31/covariance/,2010-08-31.
[2]进击的马斯特.再谈协方差矩阵之主成分分析[DB/OL].http://pinkyjie.com/2011/02/24/covariance-pca/,2011-02-24.
"""
from __future__ import division
import numpy as np

class PCAcomponent(object):
    def __init__(self, X, N=3):
        self.X = X
        self.N = N
        self.variance_ratio = []
        self.low_dataMat = []

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        covMat = np.cov(dataMat, rowvar=False)
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]
        small_eigVect = eigVect[:, eigValInd]
        self.low_dataMat = dataMat * small_eigVect
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self

class PCAcomponent_1(object):
    def __init__(self, X, N=3):
        self.X = X
        self.N = N
        self.variance_ratio = []
        self.low_dataMat = []

    def _fit(self):
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        covMat = np.cov(dataMat, rowvar=False)
        eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[:self.N]
        small_eigVect = eigVect[:, eigValInd]
        self.low_dataMat = dataMat * small_eigVect  
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self