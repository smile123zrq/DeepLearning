import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from five_layer_net import FiveLayerNet
sys.path.append(os.pardir)

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = FiveLayerNet(input_size=784, hidden1_size=50, hidden2_size=50, hidden3_size=50, hidden4_size=50, output_size=10)

iters_num = 20000   # 计算梯度，更新参数次数
train_size = x_train.shape[0]   # 训练数据数量
batch_size = 100    # mini_batch大小
learning_rate = 0.1  # 学习率

train_loss_list = []    # 损失函数值
train_acc_list = []     # 训练数据识别精度
test_acc_list = []      # 测试数据识别精度

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 从训练数据中随机抽取batch_size个数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 通过误差反向传播法计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)   # 计算损失函数
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(len(train_acc_list), '训练，测试识别精度：', train_acc, test_acc)

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

print("保存权重偏置 ...")
with open('wight.pkl', 'wb') as f:
    pickle.dump(network.params, f, -1)
print("完成!")
