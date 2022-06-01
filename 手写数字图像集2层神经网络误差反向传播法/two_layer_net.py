import sys
import os
import numpy as np
from common.layers import *
from collections import OrderedDict

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class TwoLayerNet:

    # 进行初始化，参数从头开始依次是输入层的神经元数、隐藏层的神经元数、输出层的神经元数、初始化权重时的高斯分布的规模
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}    # 保存神经网络的参数的字典型变量
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()     # 保存神经网络的层的有序字典型变量
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()   # 神经网络的最后一层,此处为SoftmaxWithLoss层

    # 进行识别（推理），参数x是图像数据
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 计算损失函数的值。
    # 参数X是图像数据、t是正确解标签（监督数据）
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 通过误差反向传播法计算关于权重参数的梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['w1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['w2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads
