# Fuction实例和Variable实例互相引用，导致循环引用，无法被垃圾回收
# 使用weakref模块解决循环引用的问题

from steps.step09 import Variable, as_array
import weakref
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0  # 记录该变量是第几代

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 设置变量的世代比创造者函数的世代大1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # 根据世代大小排序，世代小的排在前面
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()    # 1. 取出函数
            gys = [output().grad for output in f.outputs]  # 2. 获取函数的输出的梯度
            gxs = f.backward(*gys)                     # 3. 计算输入的梯度
            if not isinstance(gxs, tuple):             # 如果gxs不是元组，则将其转换为元组
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):           # 4. 将梯度设置为输入变量
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # 累加梯度

                if x.creator is not None:
                    add_func(x.creator)    # 5. 将函数的输入的创造者添加到列表中

class Function:
    # 在参数前面加*，可以在不使用列表的情况下调用具有任意个参数的函数
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  # 获取所有输入变量的data属性
        
        # 参数前添加*可以解包列表，将其中的元素作为独立的参数传递
        ys = self.forward(*xs)         # 将所有输入变量的data属性传递
        if not isinstance(ys, tuple):  # 如果输出不是元组，则将其转换为元组
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # 将所有输出变量的data属性转换为Variable对象
        
        self.generation = max([x.generation for x in inputs])  # 设置函数的世代为输入变量中最大的世代
        for output in outputs:
            output.set_creator(self)   # 设置输出变量的创造者为当前函数对象
        self.inputs = inputs          # 保存输入变量
        self.outputs = [weakref.ref(output) for output in outputs]        # 保存输出变量
        
        return outputs if len(outputs) == 1 else outputs
    # 抛出异常，告诉使用Function方法的人此方法应该通过继承来实现
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()