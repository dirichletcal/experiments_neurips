from __future__ import division
import numpy as np

from scipy.special import expit

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration

from betacal import BetaCalibration

from calib.utils.functions import fit_beta_nll
from calib.utils.functions import fit_beta_moments
from calib.utils.functions import fit_beta_midpoint
from calib.utils.functions import beta_test

from calib.utils.multiclass import OneVsRestCalibrator

from dirichlet import DirichletCalibrator
from dirichlet.calib.multinomial import MultinomialRegression

from mixture_of_dirichlet import MixDir


class _DummyCalibration(BaseEstimator, RegressorMixin):
    """Dummy Calibration model. The purpose of this class is to give
    the CalibratedClassifierCV class the option to just return the
    probabilities of the base classifier.
    """
    def fit(self, *args):
        """Does nothing"""
        return self

    def predict_proba(self, T):
        """Return the probabilities of the base classifier"""
        return T


class IsotonicCalibration(IsotonicRegression):
    def fit(self, scores, y, *args, **kwargs):
        '''
        Score=0 corresponds to y=0, and score=1 to y=1
        Parameters
        ----------
        scores : array-like, shape = [n_samples,]
            Data.
        y : array-like, shape = [n_samples, ]
            Labels.
        Returns
        -------
        self
        '''
        return super(IsotonicCalibration, self).fit(scores, y, *args, **kwargs)
    def predict_proba(self, scores, *args, **kwargs):
        return self.transform(scores, *args, **kwargs)


class SigmoidCalibration(_SigmoidCalibration):
    def predict_proba(self, *args, **kwargs):
        return super(SigmoidCalibration, self).predict(*args, **kwargs)


MAP_CALIBRATORS = {
    'uncalibrated': _DummyCalibration(),
    'isotonic': OneVsRestCalibrator(IsotonicCalibration(out_of_bounds='clip')),
    'sigmoid': OneVsRestCalibrator(SigmoidCalibration()),
    'beta': OneVsRestCalibrator(BetaCalibration(parameters="abm")),
    'beta_am': OneVsRestCalibrator(BetaCalibration(parameters="am")),
    'beta_ab': OneVsRestCalibrator(BetaCalibration(parameters="ab")),
    'ovr_dir_full': OneVsRestCalibrator(DirichletCalibrator(matrix_type='full')),
    'ovr_dir_diag': OneVsRestCalibrator(DirichletCalibrator(matrix_type='diagonal')),
    'ovr_dir_fixd': OneVsRestCalibrator(DirichletCalibrator(matrix_type='fixed_diagonal')),
    'dirichlet_full': DirichletCalibrator(matrix_type='full'),
    'dirichlet_diag': DirichletCalibrator(matrix_type='diagonal'),
    'dirichlet_fix_diag': DirichletCalibrator(matrix_type='fixed_diagonal'),
    'dirichlet_mixture': MixDir()
}


class CalibratedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, method=None, score_type=None):
        ''' Initialize a Calibrated model (classifier + calibrator)

        Parameters
        ----------
        base_estimator : string
            Name of the classifier
        method : string
            Name of the calibrator
        score_type : string
            String indicating the function to call to obtain predicted
            probabilities from the classifier.
        '''
        self.method = method
        self.base_estimator = base_estimator
        self.score_type = score_type

    def fit(self, X, y):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_classes)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         multi_output=True)
        X, y = indexable(X, y)

        scores = self.base_estimator.predict_proba(X)

        self.calibrator = clone(MAP_CALIBRATORS[self.method])
        # TODO isotonic with binary y = (n_samples, ) fails, needs one-hot-enc.
        self.calibrator.fit(scores, y)
        #print(self.method)
        #print('scores.shape(X) ' + str(scores.shape))
        #print('prob.shape(S) ' + str(self.calibrator.predict_proba(scores).shape))
        #print('prob.shape(X) ' + str(self.predict_proba(X).shape))
        #if self.method == 'isotonic':
        #    from IPython import embed; embed()
        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """

        scores = self.base_estimator.predict_proba(X)

        predictions = self.calibrator.predict_proba(scores)

        return predictions

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ["calibrator"])
        return np.argmax(self.predict_proba(X), axis=1)
