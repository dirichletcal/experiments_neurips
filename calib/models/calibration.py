from __future__ import division
import numpy as np

from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import _SigmoidCalibration as sk_sigmoid

from betacal import BetaCalibration 

from sk_calibration import _SigmoidCalibration as sk_sigmoid_notrick

from calib.utils.functions import fit_beta_nll
from calib.utils.functions import fit_beta_moments
from calib.utils.functions import fit_beta_midpoint
from calib.utils.functions import beta_test

import sys
'''
    While dirichlet isn't a package, we can import it using this relative path
'''
sys.path.insert(0,'../../dirichlet')
from dirichlet import DirichletCalibrator


class CalibratedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, method=None, score_type=None):
        if method is None:
            self.method = method
            self.platts_trick = False
        else:
            temp = method.split('-')
            self.platts_trick = (len(temp) == 2)
            self.method = temp[0]

        self.base_estimator = base_estimator
        self.score_type = score_type

    def set_base_estimator(self, base_estimator, score_type=None):
        self.base_estimator = base_estimator
        self.score_type = score_type

    def _preproc(self, X):
        if self.score_type is None:
            if hasattr(self.base_estimator, "decision_function"):
                df = self.base_estimator.decision_function(X)
                if df.ndim == 1:
                    df = df[:, np.newaxis]
            elif hasattr(self.base_estimator, "predict_proba"):
                df = self.base_estimator.predict_proba(X)
                df = df[:, 1]
            else:
                raise RuntimeError('classifier has no decision_function or '
                                   'predict_proba method.')
        else:
            if self.score_type == "sigmoid":
                df = self.base_estimator.decision_function(X)
                df = expit(df)
                if df.ndim == 1:
                    df = df[:, np.newaxis]
            else:
                if hasattr(self.base_estimator, self.score_type):
                    df = getattr(self.base_estimator, self.score_type)(X)
                    if self.score_type == "decision_function":
                        if df.ndim == 1:
                            df = df[:, np.newaxis]
                    elif self.score_type == "predict_proba":
                        df = df[:, 1:]
                else:
                    raise RuntimeError('classifier has no ' + self.score_type
                                       + 'method.')
        return df.reshape(-1)

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False)
        X, y = indexable(X, y)

        df = self._preproc(X)

        weights = None
        if self.platts_trick:
            # Bayesian priors (see Platt end of section 2.2)
            prior0 = float(np.sum(y <= 0))
            prior1 = y.shape[0] - prior0

            weights = np.zeros_like(y).astype(float)
            weights[y > 0] = (prior1 + 1.) / (prior1 + 2.)
            weights[y <= 0] = 1. / (prior0 + 2.)
            y = np.append(np.ones_like(y), np.zeros_like(y))
            weights = np.append(weights, 1.0 - weights)
            df = np.append(df, df)

        if self.method is None:
            self.calibrator = _DummyCalibration()
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'sksigmoid':
            self.calibrator = sk_sigmoid()
        elif self.method == 'sksigmoid_notrick':
            self.calibrator = sk_sigmoid_notrick()
        elif self.method == 'sigmoid':
            self.calibrator = _SigmoidCalibration()
        elif self.method == 'beta':
            self.calibrator = BetaCalibration(parameters="abm")
        elif self.method == 'beta_am':
            self.calibrator = BetaCalibration(parameters="am")
        elif self.method == 'beta_ab':
            self.calibrator = BetaCalibration(parameters="ab")
        elif self.method == 'dirichlet_full':
            self.calibrator = DirichletCalibrator()
        else:
            raise ValueError('method should be None, "sigmoid", '
                             '"isotonic", "beta", "beta_am" or "beta_ab". '
                             'Got %s.' % self.method)
        self.calibrator.fit(df, y, weights)
        # if self.method == 'beta':
        #     df_pos = df[y == 1]
        #     df_neg = df[y == 0]
        #
        #     # alpha_pos_nll, beta_pos_nll = fit_beta_nll(df_pos)
        #     # alpha_neg_nll, beta_neg_nll = fit_beta_nll(df_neg)
        #
        #     # a_nll = alpha_pos_nll - alpha_neg_nll
        #     # b_nll = beta_neg_nll - beta_pos_nll
        #     # m_nll = fit_beta_midpoint(alpha_pos_nll, beta_pos_nll,
        #     #                           alpha_neg_nll, beta_neg_nll)
        #
        #     alpha_pos_mmt, beta_pos_mmt = fit_beta_moments(df_pos)
        #     alpha_neg_mmt, beta_neg_mmt = fit_beta_moments(df_neg)
        #
        #     a_mmt = alpha_pos_mmt - alpha_neg_mmt
        #     if a_mmt < 0 or np.isnan(a_mmt):
        #         a_mmt = 0
        #     b_mmt = beta_neg_mmt - beta_pos_mmt
        #     if b_mmt < 0 or np.isnan(b_mmt):
        #         b_mmt = 0
        #     prior_pos = len(df_pos) / len(df)
        #     prior_neg = len(df_neg) / len(df)
        #     m_mmt = fit_beta_midpoint(prior_pos, alpha_pos_mmt, beta_pos_mmt,
        #                               prior_neg, alpha_neg_mmt, beta_neg_mmt)
        #     map = self.calibrator.calibrator_.map_
        # #     if a_mmt > 4 and map[0] < 2:
        # #         print [a_mmt, map[0]]
        # #         print [b_mmt, map[1]]
        # #         print [m_mmt, map[2]]
        # #         exit()
        # #     if b_mmt > 4 and map[1] < 2:
        # #         print [a_mmt, map[0]]
        # #         print [b_mmt, map[1]]
        # #         print [m_mmt, map[2]]
        # #         exit()
        #     self.a = [a_mmt, map[0]]
        #     self.b = [b_mmt, map[1]]
        #     self.m = [m_mmt, map[2]]
        #     self.df_pos = df_pos
        #     self.df_neg = df_neg
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

        proba = np.zeros((X.shape[0], 2))

        df = self._preproc(X)

        proba[:, 1] = self.calibrator.predict(df)
        proba[:, 0] = 1. - proba[:, 1]

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba

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


class _SigmoidCalibration(BaseEstimator, RegressorMixin):
    """Sigmoid regression model.

    Attributes
    ----------
    a_ : float
        The slope.

    b_ : float
        The intercept.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

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
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)
        self.lr = LogisticRegression(C=99999999999)
        self.lr.fit(X.reshape(-1, 1), y, sample_weight=sample_weight)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        return self.lr.predict_proba(T.reshape(-1, 1))[:, 1]


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

    def predict(self, T):
        """Return the probabilities of the base classifier.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        return T


class _BetaTestedCalibration(BaseEstimator, RegressorMixin):
    """Dummy regression model. The purpose of this class is to give
    the CalibratedClassifierCV class the option to just return the
    probabilities of the base classifier.


    """
    def fit(self, X, y, sample_weight=None):
        """Fit beta calibration only if the classifier is considered 
        uncalibrated.

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
        self._calibrator = BetaCalibration(parameters="abm").fit(X, y)
        test = beta_test(self._calibrator.calibrator_.map_,
                         test_type="adev", scores=X)
        if test["p-value"] >= 0.5:
            self._calibrator = _DummyCalibration().fit(X, y)
        return self

    def predict(self, T):
        """Return the probabilities of the base classifier.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        return self._calibrator.predict(T)
