#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 13:52
# !@Author  : miracleyin @email: miracleyin@live.com
# !@file: model.py

import numpy as np
import math


class FCNetwork(list):
    """
    FC network
    """

    def __init__(self, lmd=2, loss=None):
        self.loss = loss  #
        self.diff_y = None
        self.lmd = lmd

    def set_loss(self, loss):
        """
        设置网络的损失函数，运行反向传播前必须设置
        """
        self.loss = loss

    def add_layer(self, layer):
        """
        添加一层（各层按添加先后顺序组合）
        """
        layer.network = self
        layer.name += '-' + str(len(self))
        self.append(layer)

    def init(self, method):
        """
        初始化各层参数
        :param method: random、he、xavier1、xavier2、dims或normal
        """
        for layer in self:
            layer.init_prams(method)

    def forward(self, x):
        """
        前向传播所有层
        """
        if self[-1].W is None:
            raise Exception("请先运神经网络的init方法初始化各层参数")
        x = np.transpose(np.asarray(x))
        self[0].set_input(x)  # 设置输入层的输入 shape=(input_dim, data_num)
        for layer in self:
            # print(".", "*"*30, ".. ...前向"+layer.name)
            layer.forward()  # 逐层前向计算
        return np.transpose(self[-1].a)  # 最后一层的输出结果作为网络的输出 shape=(data_num, output_dim)

    def backword(self, y):
        """
        反向传播所有层
        """
        if self[-1].a is None:
            raise Exception("先运行前向传播forward")
        if self.loss is None:
            raise Exception("没有损失函数")
        y = np.transpose(np.array(y))
        y_hat = self[-1].a
        self.diff_y = self.loss.diff(y_hat, y)  # 输出的梯度
        for layer in reversed(self):
            # print(".", "*"*30, ".. ...前向"+layer.name)
            layer.backword()
        for layer in self:
            layer.greaient_descent(self.lmd)

    def next_layer(self, layer):
        """
        :param layer: 当前层
        :return: 返回当前层的下一层
        """
        if layer is self[-1]:  # 输出层无下一层
            return None
        index = self.index(layer)
        return self[index + 1]

    def prev_layer(self, layer):
        """
        :param layer: 当前层
        :return: 返回当前层的上一层
        """
        if layer is self[0]:  # 输入层无上一层
            return None
        index = self.index(layer)
        return self[index - 1]

    def get_gradient(self):
        """
        网络各层权值矩阵梯度和残差向量梯度的范数和
        """
        grad_sum = 0
        for layer in self:
            if not layer.is_input:
                grad_sum += np.linalg.norm(layer.dW) + np.linalg.norm(layer.db)
        return grad_sum

    def get_loss(self, x, y):
        """
        损失
        """
        out = self.forward(x)
        return self.loss.fun(out, np.array(y))

    def batch_generate(self, data_set, label_set, batch_size):
        """
        把数据集转成minibatch
        """
        size = len(data_set)
        data_set = np.array(data_set)
        label_set = np.array(label_set)
        num_batch = 0
        if size % batch_size == 0:
            num_batch = int(size / batch_size)
        else:
            num_batch = math.ceil(size / batch_size)
        rand_index = list(range(size))
        np.random.shuffle(list(range(size)))
        for i in range(num_batch):
            start = i * batch_size
            end = min((i + 1) * (batch_size), size)
            yield data_set[rand_index[start:end]], label_set[rand_index[start:end]]

    def train(self, data_set, label_set, dev_data, dev_label, batch_size=50, epoch=10):
        """
        训练
        """
        grads = []
        losses = []
        precs = []
        for i in range(epoch):
            j = 0
            for data_batch, label_batch in self.batch_generate(data_set, label_set, batch_size):
                j += 1
                print("... 第%d次迭代，第%d个batch" % (i, j))
                self.forward(data_batch)
                self.backword(label_batch)
            precision, grad, loss = self.validate(dev_data, dev_label)
            grads.append(grad)
            losses.append(loss)
            precs.append(precision)
            print("第 %d 次迭代，准确率 %f ，梯度 %f ，损失 %f" % (i, precision, grad, loss))
        return precs, grads, losses

    def validate(self, dev_data, dev_label):
        """
        验证
        """
        grad = self.get_gradient()
        loss = self.get_loss(dev_data, dev_label)
        precision = self.test(dev_data, dev_label)
        return precision, grad, loss

    def test(self, test_data, test_label, batch_size=512):
        """
        测试
        """
        wrong_num = 0
        # 分批测试避免测试数据量太大造成问题
        for data_batch, label_batch in self.batch_generate(test_data, test_label, batch_size):
            predict = self.forward(data_batch)
            wrong_num += np.count_nonzero(np.argmax(predict, 1) - np.argmax(label_batch, 1))
        p = 1 - wrong_num / len(test_data)
        return p
