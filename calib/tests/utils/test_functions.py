import unittest
import numpy as np
from calib.utils.functions import binary_ECE, classwise_ECE, full_ECE, MCE

from sklearn.preprocessing import label_binarize


class TestFunctions(unittest.TestCase):
    def test_binary_ece(self):
        S = np.array([.6, .6, .6, .6, .6, .6, .6, .6, .6, .6])
        y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        ece = binary_ECE(S, y)
        self.assertAlmostEqual(ece, 0)

    def test_classwise_ece(self):
        S = np.array([[0.6, 0.2, 0.2]]*10)
        Y = label_binarize([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], range(3))
        ece = classwise_ECE(S, Y)
        self.assertAlmostEqual(ece, 0)

    def test_full_ece(self):
        S = np.array([[0.6, 0.2, 0.2]]*10)
        Y = label_binarize([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], range(3))
        ece = full_ECE(S, Y)
        self.assertAlmostEqual(ece, 0)

    def test_conf_mce(self):
        S = np.ones((2, 3))/3.0
        y = np.array([0, 0])
        mce = MCE(S, y)
        self.assertAlmostEqual(mce, 2.0/3)

        y = np.array([0, 1, 2])
        S = np.array([[1/3, 0.3, 0.3],
                      [1/3, 0.3, 0.3],
                      [1/3, 0.3, 0.3]])
        mce = MCE(S, y)
        self.assertAlmostEqual(mce, 0.0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
