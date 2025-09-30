if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)  # y对x求导，得到y'

iter = 4
for i in range(iter):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)  # y'对x求导，得到y''
    print(f"{i}阶导数：{x.grad}")

gx = x.grad
gx.name = 'gx' + str(iter + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh_high_order.png')  # 可视化计算图