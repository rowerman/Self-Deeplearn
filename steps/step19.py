import numpy as np

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
    
    def __len__(self):   # len()函数调用
        return len(self.data)
    
    def __repr__(self):  # print()函数调用
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
