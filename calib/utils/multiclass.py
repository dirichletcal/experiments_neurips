# Adapted from  scikit-learn.sklearn.multiclass
import inspect
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer

from joblib import Parallel
from joblib import delayed

from sklearn.multiclass import _ConstantPredictor

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted


def _fit_binary(estimator, X, y, X_val=None, y_val=None, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        if X_val is not None and y_val is not None:
            estimator.fit(X, y, X_val=X_val, y_val=y_val)
        else:
            estimator.fit(X, y)
    return estimator


class OneVsRestCalibrator(BaseEstimator, ClassifierMixin):
    """One-vs-the-rest (OvR) multiclass/multilabel strategy

    Also known as one-vs-all, this strategy consists in fitting one calibrator
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.
    This strategy can also be used for multilabel learning, where a classifier
    is used to predict multiple labels for instance, by fitting on a 2-d matrix
    in which cell [i, j] is 1 if sample i has label j and 0 otherwise.
    In the multilabel learning literature, OvR is also known as the binary
    relevance method.
    Read more in the :ref:`User Guide <ovr_classification>`.
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    """
    def __init__(self, estimator, n_jobs=1, normalize=True):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.normalize = normalize

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """Fit underlying estimators.

        If the number of classes = 2, only one model is trained to predict the
        class 1 (second column)
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_classes]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outpreform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        if X.shape[1] == 2:
            x_columns = (X[:,1].ravel().T, )
        else:
            x_columns = (col.ravel() for col in X.T)

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        y_columns = (col.toarray().ravel() for col in Y.T)

        if 'X_val' in inspect.getargspec(self.estimator.fit).args and \
            X_val is not None:
            if X_val.shape[1] == 2:
                x_val_columns = (X_val[:,1].ravel().T, )
            else:
                x_val_columns = (col.ravel() for col in X_val.T)

            Y_val = self.label_binarizer_.transform(y_val)
            Y_val = Y_val.tocsc()
            y_val_columns = (col.toarray().ravel() for col in Y_val.T)
        else:
            x_val_columns = [None]*np.shape(Y)[0]
            y_val_columns = [None]*np.shape(Y)[0]

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, x_column, y_column, x_val_column, y_val_column,
            classes=[ "not %s" % self.label_binarizer_.classes_[i],
                     self.label_binarizer_.classes_[i]])
            for i, (x_column, y_column, x_val_column, y_val_column) in enumerate(zip(x_columns, y_columns, x_val_columns,
                                                         y_val_columns)))

        return self

    @if_delegate_has_method(['_first_estimator', 'estimator'])
    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by label of classes.
        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.
        In the single label multiclass case, the rows of the returned matrix
        sum to 1.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, 'estimators_')
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        if X.shape[1] == 2:
            x_columns = (X[:,1].ravel().T, )
        else:
            x_columns = (col.ravel() for col in X.T)

        # Removed indexing as follows: e.predict_proba(x_column)[:, 1]
        Y = np.array([e.predict_proba(x_column)
                      for (e, x_column) in zip(self.estimators_, x_columns)]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
            # Change all columns to zero for a uniform prediction
            Y[np.isnan(Y)] = 1/Y.shape[1]

        return Y

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier"""
        return self.label_binarizer_.y_type_.startswith('multilabel')
