#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/19 上午10:54
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : activater.py

import numpy as np


class Sigmoid(object):
    """
    Sigmoid激活函数
    """

    @staticmethod
    def fun(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def diff(x):
        return Sigmoid.fun(x) * (1 - Sigmoid.fun(x))


class Tanh(object):
    """
    tanh双曲正切激活函数
    """

    @staticmethod
    def fun(x):
        return np.tanh(x)

    @staticmethod
    def diff(x):
        return 1 - Tanh.fun(x) * Tanh.fun(x)


class Relu(object):
    """
    Relu激活函数
    leaky Relu
    """

    @staticmethod
    def fun(x):
        return (x > 0) * x + (x <= 0) * (0.01 * x)

    @staticmethod
    def diff(x):
        grad = 1. * (x > 0)
        grad[grad == 0] = 0.01
        return grad


if __name__ == '__main__':
    a = Sigmoid()
    print(a.diff(2))
