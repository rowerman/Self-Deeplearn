class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

import numpy as np

data = np.array(1.0)
x = Variable(data)
print(x.data)  # Should output: 1.0