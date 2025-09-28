from steps.step12 import Function
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()    # 1. 取出函数
            gys = [output.grad for output in f.outputs]  # 2. 获取函数的输出的梯度
            gxs = f.backward(*gys)                     # 3. 计算输入的梯度
            if not isinstance(gxs, tuple):             # 如果gxs不是元组，则将其转换为元组
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):           # 4. 将梯度设置为输入变量
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)    # 5. 将函数的输入的创造者添加到列表中

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        # 虽然输入修改为了多个，但这里仍然只有一个输入
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx