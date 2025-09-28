from steps.step07 import Function
import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)
    
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
            if f is not None:
                x, y = f.input, f.output  # 2. 获取函数的输入和输出
                x.grad = f.backward(self.grad)

                if x.creator is not None:
                    funcs.append(x.creator)    # 3. 将函数的输入的创造者添加到列表中

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        # 当前函数是输出的创造者，用于实现自动化反向传播
        output.set_creator(self)
        self.input = input
        self.output = output  # 记录输出变量
        return output
    
    # 抛出异常，告诉使用Function方法的人此方法应该通过继承来实现
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()