# 扩展Variable类，使Variable实例能够与ndarray实例，以及int和float等数值类型进行加法和乘法运算
import weakref
import numpy as np
from steps.step18 import Config

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name     # 用于区分变量 
        self.grad = None
        self.creator = None
        self.generation = 0  # 记录该变量是第几代
        self.__array_priority__ = 200  # ndarray的优先级是0，Variable的优先级更高

    def __len__(self):   # len()函数调用
        return len(self.data)
    
    def __repr__(self):  # print()函数调用
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 设置变量的世代比创造者函数的世代大1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
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
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y是弱引用的对象，必须通过()获取原始对象

    @property   # 使得shape方法就可以作为实例变量被访问
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):     # 维度
        return self.data.ndim
    
    @property
    def size(self):     # 元素个数
        return self.data.size
    
    @property
    def dtype(self):    # 数据类型
        return self.data.dtype
    
class Function:
    # 在参数前面加*，可以在不使用列表的情况下调用具有任意个参数的函数
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # 确保inputs中的元素都是Variable类型

        xs = [x.data for x in inputs]  # 获取所有输入变量的data属性
        
        # 参数前添加*可以解包列表，将其中的元素作为独立的参数传递
        ys = self.forward(*xs)         # 将所有输入变量的data属性传递
        if not isinstance(ys, tuple):  # 如果输出不是元组，则将其转换为元组
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # 将所有输出变量的data属性转换为Variable对象
        
        if Config.enable_backprop:  # 如果启用反向传播，则设置相关属性
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
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        # 虽然输入修改为了多个，但这里仍然只有一个输入
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def mul(x0, x1):
    return Mul()(x0, x1)

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

Variable.__add__ = add
Variable.__mul__ = mul
Variable.__radd__ = add
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow