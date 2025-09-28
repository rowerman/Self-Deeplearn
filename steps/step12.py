from steps.step09 import Variable, as_array

class Function:
    # 在参数前面加*，可以在不使用列表的情况下调用具有任意个参数的函数
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]  # 获取所有输入变量的data属性
        
        # 参数前添加*可以解包列表，将其中的元素作为独立的参数传递
        ys = self.forward(*xs)         # 将所有输入变量的data属性传递
        if not isinstance(ys, tuple):  # 如果输出不是元组，则将其转换为元组
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # 将所有输出变量的data属性转换为Variable对象
        
        for output in outputs:
            output.set_creator(self)   # 设置输出变量的创造者为当前函数对象
        self.inputs = inputs          # 保存输入变量
        self.outputs = outputs        # 保存输出变量
        
        return outputs if len(outputs) == 1 else outputs
    # 抛出异常，告诉使用Function方法的人此方法应该通过继承来实现
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
def add(x0, x1):
    return Add()(x0, x1)