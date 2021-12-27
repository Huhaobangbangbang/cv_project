"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2021/12/3 10:23 PM
"""
import numpy as np
# 设置参数，来实现学习率递减
n_epochs = 50
t0, t1 = 5, 50

# 学习率迭代函数
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        # 随机选择一条样本
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
