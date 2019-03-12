import unittest
import numpy as np
from . import get_simple_ternary_example
from calib.utils.functions import binary_ECE, classwise_ECE, full_ECE

from sklearn.metrics import accuracy_score


class TestFunctions(unittest.TestCase):
    def test_binary_ece(self):
        S, Y = get_simple_ternary_example()
        print(binary_ECE(S, Y))

    def test_classwise_ece(self):
        S, Y = get_simple_ternary_example()
        print(classwise_ECE(S, Y))

    def test_full_ece(self):
        S, Y = get_simple_ternary_example()
        print(full_ECE(S, Y))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
