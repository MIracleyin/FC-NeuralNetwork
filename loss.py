#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/19 上午10:55
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : loss.py

import numpy as np
import math
import pickle

class CrossEntropyWithSoftmax(object):
    """
    带softmax的交叉熵损失函数
    """
    @staticmethod
    def fun(y_hat, y):
        yr_hot = CrossEntropyWithSoftmax.softmax(y_hat) * y
        return np.average(- np.log(np.sum(yr_hot, 1)))

    @staticmethod
    def diff(y_hat, y):
        return y_hat - y

    @staticmethod
    def softmax(y_hat):
        e_x = np.exp(y_hat - np.max(y_hat, 0))
        return e_x / e_x.sum(0)


class MSELoss(object):
    """
    均方误差损失函数
    """
    @staticmethod
    def fun(y_hat, y):
        l = sum(np.average(0.5*(y_hat - y)*(y_hat - y), 0))
        return l

    @staticmethod
    def diff(y_hat, y):
        return y_hat - y


if __name__ == '__main__':
    a = Sigmoid()
    print(a.diff(2))
