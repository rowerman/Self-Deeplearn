# 泰勒展开的导数
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import math

def my_sin(x, threshold=1e-5):
    y = 0
    for i in range(10000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x = Variable(np.array(np.pi / 4))
y = my_sin(x)
y.backward()
x.name = 'x'
y.name = 'y'

print(y.data)
print(x.grad)
plot_dot_graph(y, verbose=True, to_file='my_sin.png')