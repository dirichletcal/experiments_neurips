from __future__ import division

from sklearn.base import BaseEstimator


class MockClassifier(BaseEstimator):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, X, *args, **kwargs):
        return X

    def predict_proba(self, X, *args, **kwargs):
        return X
