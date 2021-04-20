#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/4/19 上午10:47
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : utils.py

import numpy as np
import math
import pickle

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

