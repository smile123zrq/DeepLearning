import numpy as np


def step_function(x):
    """
    激活函数--阶跃函数
    :param x:NumPy数组
    :return:x中大于0则输出1，否则输出0

    x = np.array([-1.0, 1.0, 2.0])
    y = x > 0
    y --> array([False, True, True], dtype=bool)
    对NumPy数组进行不等号运算后，数组的各个元素都会进行不等号运算，生成一个布尔型数组。
    x中大于0的元素替换成True，小于等于0的元素替换成False。
    阶跃函数会输出int型的0或1，所以需要将数组y的元素类型转换为int型
    y = y.astype(int) --> array([0, 1, 1])
    """
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    """
    激活函数--sigmoid函数
    :param x: NumPy数组
    :return:输出0-1的连续值
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    激活函数--ReLU函数
    :param x: NumPy数组
    :return: 当x大于0时，直接输出该值，当x小于等于0时，输出0

    np.maximum()函数会从输入的数值中选择较大的那个进行输出
    """
    return np.maximum(0, x)


def identity_function(x):
    """
    输出层激活函数--恒等函数
    :param x:NumPy数组
    :return:返回x本身
    """
    return x


def softmax(x):
    """
    输出层激活函数--softmax函数
    :param x: NumPy数组
    :return:输出0-1之间的实数值

    softmax函数的输出是0.0到1.0之间的实数。并且，softmax函数的输出值的总和是1。
    所以可以将softmax函数的输出解释为概率。
    """
    if x.ndim == 2:  # x为二维，矩阵
        x = x.T  # x变为x转置
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策,减去x中最大的数字
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    """
    损失函数--交叉熵误差
    :param y: 模型的识别结果y
    :param t: 数据标签，监督数据
    :return: 交叉熵误差函数的值

    mini-batch版交叉熵误差，可以同时处理单个数据和批量数据
    np.arange(batch_size)会生成一个从0 到batch_size-1的数组。
    np.arange(5)会生成一个NumPy数组[0, 1, 2, 3, 4]
    t中标签是以[2, 7, 0, 9, 4]的形式存储的，所以y[np.arange(batch_size),t]
    能抽出各个数据的正确解标签对应的神经网络的输出（在这个例子中，y[np.arange(batch_size), t]
    会生成NumPy 数组[y[0,2], y[1,7], y[2,0],y[3,9], y[4,4]]）
    表示取y中第0行取第2处元素，第1行取第7处元素
    """
    if y.ndim == 1:  # y的维度为1，即单个数据，需要改变数据的形状    [1, 2, 3]
        t = t.reshape(1, t.size)  # 转化为[[1, 2, 3]]
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引，也就是 one-hot 表示
    # 将监督数据转化为标签形式，非ont-hot表示
    if t.size == y.size:
        t = t.argmax(axis=1)    # 返回矩阵每一行中最大值的索引

    batch_size = y.shape[0]     # 用于正规化，通过除以batch_size，可以求单个数据的“平均损失函数”
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def mean_squared_error(y, t):
    """
    损失函数--均方误差
    :param y: 模型的识别结果y
    :param t: 数据标签，监督数据
    :return: 均方误差函数的值
    t为one-hot表示的数据
    """
    return 0.5 * np.sum((y - t) ** 2)
