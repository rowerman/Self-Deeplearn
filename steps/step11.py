from steps.step09 import Variable, as_array

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]  # 获取所有输入变量的data属性
        ys = self.forward(xs)         # 将所有输入变量的data属性传递
        outputs = [Variable(as_array(y)) for y in ys]  # 将所有输出变量的data属性转换为Variable对象
        for output in outputs:
            output.set_creator(self)   # 设置输出变量的创造者为当前函数对象
        self.inputs = inputs          # 保存输入变量
        self.outputs = outputs        # 保存输出变量
        return outputs   
    # 抛出异常，告诉使用Function方法的人此方法应该通过继承来实现
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)