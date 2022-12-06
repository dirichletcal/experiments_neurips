from sklearn.preprocessing import LabelBinarizer
import numpy as np

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, *args, **kwargs):
        y_bin = super().transform(*args, **kwargs)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1-y_bin, y_bin])
        print(y_bin.shape)
        return y_bin

    def fit_transform(self, *args, **kwargs):
        y_bin = super().fit_transform(*args, **kwargs)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1-y_bin, y_bin])
        print(y_bin.shape)
        return y_bin
