#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/19 上午10:58
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : module.py
import numpy as np
import math
import pickle



class Linear(object):
    """
    神经网络的一层
    """

    def __init__(self, node_num, activate_fun=None, is_input=False):
        if activate_fun is None and is_input is False:
            raise Exception("非输入层必须指定激活函数")
        self.dim = node_num
        self.W = None  # 权值矩阵
        self.dW = None  # 权值矩阵梯度
        self.b = None  # 残差向量
        self.db = None  # 残差向量梯度
        self.z = None  # 当前层的总输入 z=Wa_p +b
        self.a = None  # 当前层前向计算的输出向量 a=activate(z)
        self.delta = None  # 反向传播的delta，即 dC/dz
        self.activate = activate_fun
        self.is_input = is_input  # 当前层是否是输入层
        self.network = None  # 当前层所属的神经网络
        self.prev_layer = None  # 当前层的前一层
        self.next_layer = None  # 当前层的后一层
        self.name = "Linear"

    def set_input(self, x):
        """
        输入层的输出
        """
        if self.is_input:  # 只有输入层能输入
            self.x = x

    def forward(self):
        """
        前向传播
        """
        if self.is_input:
            self.a = self.x  # 如果是输入层，那么输入激活函数层设为输入
            return
        self.z = np.dot(self.W, self.prev_layer.a) + self.b  # z = Wa^[l-1] + b; shape=(dim, data_num) 输出为权重和上一层点乘加偏置
        self.a = self.activate.fun(self.z)  # 通过激活函数

    def backword(self):
        """
        反向传播
        """
        if self.is_input:
            return  # 输入层无任何操作
        if self is self.network[-1]:  # 若为输出层
            self.delta = self.activate.diff(self.z) * self.network.diff_y  # delta=sigma'(z) * dy; shape=(dim,data_num)
        else:  # 若不是输出层
            W_next = self.next_layer.W  # 获取下一层的权重
            trans_expend_next_delta = np.expand_dims(np.transpose(self.next_layer.delta), 2)  # 转置后
            W_next_delta_next = np.matmul(np.transpose(W_next), trans_expend_next_delta)
            # a * mul(W^[l-1], delta^[l-1])
            self.delta = self.activate.diff(self.z) * np.transpose(np.squeeze(W_next_delta_next, 2))

        # 求参数梯度
        delta_expand = np.expand_dims(np.transpose(self.delta), 2)  # 改变形状以适于批量矩阵运算
        prev_a_expand = np.expand_dims(np.transpose(self.prev_layer.a), 1)  # 改变形状以适于批量矩阵运算
        self.dW = np.average(np.matmul(delta_expand, prev_a_expand),
                             0)  # dW=mul(delta,a^[l-1]); shape=(dim,dim^[l-1]) todo: shape
        self.db = np.expand_dims(np.average(self.delta, 1), 1)  # db=delta ; shape=(dim,1)
        self.clip_gradient()  # clipse gradient，防止梯度爆炸

    def clip_gradient(self):
        """
        避免梯度爆炸
        """
        threshold = 1 / self.network.lmd  # 最大梯度设置成1/lr
        norm_dW = np.linalg.norm(self.dW)
        norm_db = np.linalg.norm(self.db)
        if norm_dW > threshold:
            self.dW = threshold * self.dW / norm_dW
            print("... ... 权值矩阵梯度 cliped!")
        if norm_db > threshold:
            self.db = threshold * self.db / norm_db
            print("... ... 权值矩阵梯度 cliped!")

    def greaient_descent(self, lmd):
        if self.is_input:  # 输入层无参数更新
            return
            # 梯度下降更新参数
        self.W = self.W - lmd * self.dW
        self.b = self.b - lmd * self.db

    def init_prams(self, method):
        """
        随机初始化权值矩阵和残差向量，确定当前层的前一层和后一层
        :param method: random、he、xavier1、xavier2、dims或normal
        """
        self.prev_layer = self.network.prev_layer(self)  # prev
        self.next_layer = self.network.next_layer(self)  # next

        if self.is_input:  # 输入层无权值矩阵和残差向量
            return

        if self.W is not None and self.b is not None:  # 如果W和b已存在，则不用再随机初始化
            return
        self.b = np.zeros(shape=[self.dim, 1])  # 初始化偏置向量为0向量
        # 多种权值初始化方法
        if method == "random":
            self.W = np.random.randn(self.dim, self.prev_layer.dim) * 0.01
        elif method == "he":
            self.W = np.random.randn(self.dim, self.prev_layer.dim) * np.sqrt(2 / self.prev_layer.dim) * .01
        elif method == "xavier1":
            self.W = np.random.randn(self.dim, self.prev_layer.dim) * np.sqrt(1 / self.prev_layer.dim) * .01
        elif method == "xavier2":
            bound = np.sqrt(6 / (self.dim + self.prev_layer.dim))  # 6/sqrt(dim + pre_dim)
            self.W = np.random.uniform(-bound, bound, size=[self.dim, self.prev_layer.dim])
        elif method == "dims":
            bound = np.sqrt(6 / (self.dim + self.prev_layer.dim))  # 6/sqrt(dim + pre_dim)
            self.W = np.random.uniform(-bound, bound, size=[self.dim, self.prev_layer.dim])
        elif method == "normal":
            self.W = np.random.normal(size=[self.dim, self.prev_layer.dim])  # 标准正态分布初始化

    def set_params(self, W, b):

        """
        手动设置权值矩阵和残差向量
        """
        if self.is_input:
            raise ("输入层无权值矩阵和残差向量")
        self.W = np.array(W)
        self.b = np.array(b)


def batch_generate(data_set, label_set, batch_size):
    """
    把数据集转成minibatchÒ
    """
    size = len(data_set)
    data_set = np.array(data_set)
    label_set = np.array(label_set)
    num_batch = 0
    if size % batch_size == 0:
        num_batch = int(size/batch_size)
    else:
        num_batch = math.ceil(size/batch_size)
    rand_index = list(range(size))
    np.random.shuffle(list(range(size)))
    for i in range(num_batch):
        start = i*batch_size
        end = min((i+1)*(batch_size), size)
        yield data_set[rand_index[start:end]], label_set[rand_index[start:end]]


def save_model(model, path):
    """
    保存模型
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    加载模型
    """
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == '__main__':
    x = np.random.rand(10, 1)

    pass
