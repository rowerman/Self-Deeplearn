# 使用牛顿法进行优化
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable

def f(x):
    return x**4 - 2*x**2

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)  # 计算一阶导数

    gx = x.grad
    x.cleargrad()
    gx.backward()                  # 计算二阶导数
    gx2 = x.grad

    x.data -= gx.data / gx2.data  # 牛顿法更新
