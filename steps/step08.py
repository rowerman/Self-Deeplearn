# 递归变循环

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()    # 1. 取出函数
            if f is not None:
                x, y = f.input, f.output  # 2. 获取函数的输入和输出
                x.grad = f.backward(self.grad)

                if x.creator is not None:
                    funcs.append(x.creator)    # 3. 将函数的输入的创造者添加到列表中