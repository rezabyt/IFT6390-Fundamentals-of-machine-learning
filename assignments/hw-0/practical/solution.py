import numpy as np


def make_array_from_list(some_list):
    return np.array(some_list)


def make_array_from_number(num):
    return np.array([num])


class NumpyBasics:
    def add_arrays(self, a, b):
        return np.array(a) + np.array(b)

    def add_array_number(self, a, num):
        return np.array(a) + num

    def multiply_elementwise_arrays(self, a, b):
        return np.multiply(np.array(a), np.array(b))

    def dot_product_arrays(self, a, b):
        return np.dot(np.array(a), np.array(b))

    def dot_1d_array_2d_array(self, a, m):
        # consider the 2d array to be like a matrix
        return np.matmul(np.array(a), np.array(m))
