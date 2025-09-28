import unittest
import numpy as np
from steps.step04 import numerical_diff
from steps.step09 import Variable, Function, as_array, square

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertTrue(np.allclose(x.grad, expected))

    def test_gradients(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        self.assertTrue(np.allclose(x.grad, num_grad))