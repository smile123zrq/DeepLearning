from common.functions import *


class Affine:
    """
    Affine层，负责进行矩阵乘法
    考虑了输入数据为张量（四维数据）的情况
    """
    def __init__(self, w, b):
        self.w = w  # 权重
        self.b = b  # 转置

        self.x = None   # 输入x
        self.original_x_shape = None    # 原始的x的形状
        # 权重和偏置参数的导数
        self.dw = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape     # 记录四维数据的形状
        x = x.reshape(x.shape[0], -1)   # 将这个四维数据转化为二维的矩阵，方便进行矩阵运算
        '''
         当 X.shape  为  (209, 64, 64, 3)
         X.reshape(X.shape[0], -1)   变为   (209, 64*64*3) 
         X.reshape(X.shape[0], -1).T  变为  (64*64*3, 209)
         这样就可以将多维数据转变成二维的矩阵，这样就可以使用矩阵乘法直接进行运算了
        '''
        self.x = x

        out = np.dot(self.x, self.w) + self.b   # 矩阵乘法，权重偏置加权和

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 将上面变化为矩阵的多维数据还原
        return dx   # 之所以这里只返回dx是因为它的下一个节点的计算需要用到dx,,而不需要dw与db


class ReLU:
    """
    激活函数RelU层，负责使用激活函数ReLU
    """
    def __init__(self):
        # mask变量由True/False构成的NumPy数组，它会把正向传播时输入的x的元素中的小于等于0的地方保存为True，
        # 其他地方（大于0的地方）保存为False
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # (x <= 0)这个不等式会自动将数组中小于等于0的元素变为True,大于0的变为False
        out = x.copy()  # 使用x的副本，这样就不会改变x中的原有数据
        out[self.mask] = 0  # 以mask作为索引，这样mask中为True的地方变为0，为False的地方变为原来的值

        return out

    def backward(self, dout):
        # 将mask中为True的地方变为0，为False的地方变为原来的值，
        # 这样就实现了，当x大于0时原样返回上游传过来的导数，当x小于0时，就像下游传输0
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """
    激活函数Sigmoid层
    """
    def __init__(self):
        self.out = None  # 保存正向传播的输出 out，反向传播时，使用该变量进行计算

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class SoftmaxWithLoss:
    """
    输出层，Softmax函数与交叉熵误差（计算损失函数）
    """
    def __init__(self):
        self.loss = None    # 损失函数计算得出的损失
        self.y = None  # softmax的输出，模型算出的结果y
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        # 传过来的参数是模型直接识别的结果，没有使用输出层的激活函数转换，因为要就行学习，这里使用softmax计算出y来计算损失函数
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)     # 使用交叉熵误差计算出当前损失函数的值

        return self.loss    # 返回的是最后的损失函数的值

    def backward(self):
        batch_size = self.t.shape[0]    # 用于正规化，通过除以batch_size，可以求单个数据的“平均导数”
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # 将与dx中与t对应的位置上减1
            dx = dx / batch_size

        return dx   # 返回
