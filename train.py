#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/20 14:03
# !@Author  : miracleyin @email: miracleyin@live.com
# !@file: train.py.py
from data_tools import load_iris_data
from module import Linear
from model import FCNetwork
from activater import Tanh
from loss import MSELoss


def main():
    # 加载数据
    dataset = load_iris_data('iris.data.txt')
    x = [data[0] for data in dataset]
    y = [data[1] for data in dataset]

    x_train = x[0: 120]
    y_train = y[0: 120]
    x_test = x[120:]
    y_test = y[120:]

    # 创建模型
    net = FCNetwork(0.1)
    act_fun = Tanh
    loss_fun = MSELoss

    # 线性层
    input_layer = Linear(4, is_input=True)
    net.add_layer(input_layer)
    hidden1 = Linear(20, activate_fun=act_fun)
    net.add_layer(hidden1)
    hidden2 = Linear(10, activate_fun=act_fun)
    net.add_layer(hidden2)
    output_layer = Linear(3, activate_fun=act_fun)
    net.add_layer(output_layer)

    net.init('dims')
    net.set_loss(loss_fun)
    # 训练
    precs, grad, loss = net.train(x_train, y_train, x_test, y_test, 30, 500)
    # 测试
    p = net.test(x_test, y_test)
    print("------\n测试准确率", p)

    # 画图显示结果
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体，否则中文乱码

    plt.subplot(3, 1, 1)  # 上图，2行2列第1幅图
    plt.title("Test Precision:" + str(p), fontsize=15)
    plt.plot(precs, color='g', label="precision")
    plt.legend(loc='upper right', frameon=True)

    plt.subplot(3, 1, 2)  # 上图，2行2列第1幅图
    plt.plot(grad, color='r', label="gradient")
    plt.legend(loc='upper right', frameon=True)

    plt.subplot(3, 1, 3)  # 下图，2行2列第2幅图
    plt.plot(loss, color='b', label="loss")
    plt.legend(loc='upper right', frameon=True)

    plt.show()


if __name__ == '__main__':
    main()
