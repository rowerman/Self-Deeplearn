import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    # 1. 获取函数
        if f is not None:
            x = f.input     # 2. 获取函数的输入
            x.grad = f.backward(self.grad)
            x.backward()    # 3. 调用自己前面那个变量的backward方法

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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