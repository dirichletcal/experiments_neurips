from __future__ import division
import numpy as np

from scipy.special import expit

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

class IsotonicCalibration(IsotonicRegression):
    def predict_proba(self, *args, **kwargs):
        return super(IsotonicCalibration, self).predict(*args, **kwargs)

class SigmoidCalibration(_SigmoidCalibration):
    def predict_proba(self, *args, **kwargs):
        return super(SigmoidCalibration, self).predict(*args, **kwargs)

class CalibratedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, method=None, score_type=None):
        self.method = method
        self.base_estimator = base_estimator
        self.score_type = score_type

    def set_base_estimator(self, base_estimator, score_type=None):
        self.base_estimator = base_estimator
        self.score_type = score_type

    def _preproc(self, X):
        # if self.score_type is None:
        #     if hasattr(self.base_estimator, "decision_function"):
        #         df = self.base_estimator.decision_function(X)
        #         if df.ndim == 1:
        #             df = df[:, np.newaxis]
        #     elif hasattr(self.base_estimator, "predict_proba"):
        df = self.base_estimator.predict_proba(X)
        #         df = df[:, 1]
        #     else:
        #         raise RuntimeError('classifier has no decision_function or '
        #                            'predict_proba method.')
        # else:
        #     if self.score_type == "sigmoid":
        #         df = self.base_estimator.decision_function(X)
        #         df = expit(df)
        #         if df.ndim == 1:
        #             df = df[:, np.newaxis]
        #     else:
        #         if hasattr(self.base_estimator, self.score_type):
        #             df = getattr(self.base_estimator, self.score_type)(X)
        #             if self.score_type == "decision_function":
        #                 if df.ndim == 1:
        #                     df = df[:, np.newaxis]
        #             elif self.score_type == "predict_proba":
        #                 df = df[:, 1:]
        #         else:
        #             raise RuntimeError('classifier has no ' + self.score_type
        #                                + 'method.')
        # return df.reshape(-1)
        return df

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_classes)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         multi_output=True)
        X, y = indexable(X, y)

        df = self._preproc(X)

        # weights = None
        # if self.platts_trick:
        #     # Bayesian priors (see Platt end of section 2.2)
        #     prior0 = float(np.sum(y <= 0))
        #     prior1 = y.shape[0] - prior0

        #     weights = np.zeros_like(y).astype(float)
        #     weights[y > 0] = (prior1 + 1.) / (prior1 + 2.)
        #     weights[y <= 0] = 1. / (prior0 + 2.)
        #     y = np.append(np.ones_like(y), np.zeros_like(y))
        #     weights = np.append(weights, 1.0 - weights)
        #     df = np.append(df, df)

        if self.method is None:
            self.calibrator = _DummyCalibration()
        elif self.method == 'isotonic':
            self.calibrator = OneVsRestCalibrator(IsotonicCalibration(out_of_bounds='clip'))
        elif self.method == 'sigmoid':
            self.calibrator = OneVsRestCalibrator(SigmoidCalibration())
        elif self.method == 'beta':
            self.calibrator = OneVsRestCalibrator(BetaCalibration(parameters="abm"))
        elif self.method == 'beta_am':
            self.calibrator = OneVsRestCalibrator(BetaCalibration(parameters="am"))
        elif self.method == 'beta_ab':
            self.calibrator = OneVsRestCalibrator(BetaCalibration(parameters="ab"))
        elif self.method == 'multinomial':
            self.calibrator = MultinomialRegression()
        elif self.method == 'dirichlet_full':
            self.calibrator = DirichletCalibrator(matrix_type='full')
        elif self.method == 'dirichlet_diag':
            self.calibrator = DirichletCalibrator(matrix_type='diagonal')
        elif self.method == 'dirichlet_fix_diag':
            self.calibrator = DirichletCalibrator(matrix_type='fixed_diagonal')
        else:
            raise ValueError('method should be None, "multinomial", '
                             'or "dirichlet_full". '
                             'Got %s.' % self.method)
        self.calibrator.fit(df, y, sample_weight)
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

        df = self._preproc(X)

        # FIXME Should this be predict or predict_proba?
        # if hasattr(self.calibrator, "predict_proba"):
        predictions = self.calibrator.predict_proba(df)
        # elif hasattr(self.calibrator, "predict"):
        #     prediction = self.calibrator.predict(df)
        # else:
	    #     raise RuntimeError('classifier has no predict_proba or ' +
	    #                        'predict method.')

        # if len(prediction.shape) == 1:
        #     proba[:, 1] = prediction
        #     proba[:, 0] = 1. - proba[:, 1]
        # elif len(prediction.shape) == 2:
        #     proba = prediction
        # else:
        #     raise RuntimeError('The prediction is expected to be binary ' +
        #                        'but multiple classes returned from calibrator')

        # Deal with cases where the predicted probability minimally exceeds 1.0
        # proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

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


class _DummyCalibration(BaseEstimator, RegressorMixin):
    """Dummy regression model. The purpose of this class is to give
    the CalibratedClassifierCV class the option to just return the
    probabilities of the base classifier.


    """
    def fit(self, X, y, sample_weight=None):
        """Does nothing.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        return self

    def predict_proba(self, T):
        """Return the probabilities of the base classifier.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples, n_classes)
            The predicted data.
        """
        return T
