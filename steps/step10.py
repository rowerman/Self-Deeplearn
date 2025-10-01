import unittest
from steps.step09 import Variable, Function, square, exp
from steps.step04 import numerical_diff
import numpy as np

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.grad = np.array(1.0)
        y.backward()
        expected_grad = 6.0  # dy/dx at x=3 is 2*x = 6
        self.assertEqual(x.grad, expected_grad)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        # 检验两个ndarray实例的值是否相近
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)