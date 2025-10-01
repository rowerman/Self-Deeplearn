if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)  # y对x求导，得到y'
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()  # z对x求导
print(x.grad)  # z对x求导的结果