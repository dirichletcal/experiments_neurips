from __future__ import division
import numpy as np

from scipy.special import expit

from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from sklearn.metrics import log_loss

from betacal import BetaCalibration

from calib.utils.functions import fit_beta_nll
from calib.utils.functions import fit_beta_moments
from calib.utils.functions import fit_beta_midpoint
from calib.utils.functions import beta_test

from calib.utils.multiclass import OneVsRestCalibrator

from dirichlet import DirichletCalibrator
from dirichlet.calib.gendirichlet import GenerativeDirichletCalibrator
from dirichlet.calib.multinomial import MultinomialRegression
from dirichlet.calib.fixeddirichlet import FixedDiagonalDirichletCalibrator
#from dirichlet.calib.typeiidirichlet import TypeIIDirichletCalibrator
from dirichlet.calib.vectorscaling import VectorScaling
from dirichlet.calib.tempscaling import TemperatureScaling
from dirichlet.calib.matrixscaling import MatrixScaling
from dirichlet.calib.fulldirichlet import FullDirichletCalibrator

# from mixture_of_dirichlet import MixDir

from .dirichlet_keras import Dirichlet_NN


class _DummyCalibration(BaseEstimator, RegressorMixin):
    """Dummy Calibration model. The purpose of this class is to give
    the CalibratedClassifierCV class the option to just return the
    probabilities of the base classifier.
    """
    def fit(self, *args, **kwargs):
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

def logit(x):
    eps = np.finfo(x.dtype).eps
    x = np.clip(x, eps, 1-eps)
    return np.log(x/(1 - x))

def log_encode(x):
    eps = np.finfo(x.dtype).eps
    x = np.clip(x, eps, 1)
    return np.log(x)

class LogisticCalibration(LogisticRegression):
    def __init__(self, C=1.0, solver='lbfgs', multi_class='multinomial',
                 log_transform=True):
        self.C_grid = C
        self.C = C if isinstance(C, float) else C[0]
        self.solver = solver
        self.log_transform = log_transform
        self.encode = log_encode if  log_transform else logit
        self.multiclass=multi_class
        super(LogisticCalibration, self).__init__(C=C, solver=solver,
                                                  multi_class=multi_class)

    def fit(self, scores, y, X_val=None, y_val=None, *args, **kwargs):
        if isinstance(self.C_grid, list):
            calibrators = []
            losses = np.zeros(len(self.C_grid))
            for i, C in enumerate(self.C_grid):
                cal = LogisticCalibration(C=C, solver=self.solver,
                                          multi_class=self.multi_class,
                                          log_transform=self.log_transform)
                cal.fit(scores, y)
                losses[i] = log_loss(y_val, cal.predict_proba(X_val))
                calibrators.append(cal)
            best_idx = losses.argmin()
            self.C = calibrators[best_idx].C
        return super(LogisticCalibration, self).fit(self.encode(scores), y,
                *args, **kwargs)

    def predict_proba(self, scores, *args, **kwargs):
        return super(LogisticCalibration,
                self).predict_proba(self.encode(scores), *args, **kwargs)

    def predict(self, scores, *args, **kwargs):
        return super(LogisticCalibration, self).predict(self.encode(scores),
                *args, **kwargs)


class SigmoidCalibration(_SigmoidCalibration):
    def predict_proba(self, *args, **kwargs):
        return super(SigmoidCalibration, self).predict(*args, **kwargs)


class BinningCalibration(BaseEstimator, RegressorMixin):
    def __init__(self, n_bins=10, strategy='uniform', alpha=1.0):
        '''
        alpha : Laplace smoothing (x + a)/(N + 2a)
        n_bins: Number of bins
        stragegy:
                - uniform: for equal width bins
                - quantile: for equal frequency bins
                - kmeans: for each bin with same nearest center to a 1D k-means
        '''
        self.strategy = strategy
        self.n_bins = n_bins
        self.n_bins_grid = n_bins
        self.alpha = alpha

    def fit(self, scores, y, X_val=None, y_val=None, *args, **kwargs):
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
        if isinstance(self.n_bins, list):
            if X_val is None or y_val is None:
                raise ValueError('If n_bins is a list, scores_val and y_val are required during fit')
            calibrators = []
            losses = []
            for n_bins in self.n_bins:
                cal = BinningCalibration(n_bins=n_bins, strategy=self.strategy,
                                           alpha=self.alpha)
                cal.fit(scores, y)
                predict = cal.predict_proba(X_val)
                losses.append(log_loss(y_val, predict))
                calibrators.append(cal)
            best_idx = np.argmin(losses)
            self.n_bins = calibrators[best_idx].n_bins
            self.bins = calibrators[best_idx].bins
            self.predictions = calibrators[best_idx].predictions
            return self

        if len(np.shape(scores)) > 1:
            scores = scores[:,1]
        # TODO check that this code is correct:
        if self.strategy == 'quantile':
            self.bins = np.sort(scores)[::int(np.ceil(len(scores)/self.n_bins))]
            self.bins = np.hstack([self.bins, scores[-1]])
        elif self.strategy == 'uniform':
            self.bins = np.linspace(scores.min(), scores.max(), self.n_bins+1)
        self.bins[0] = - np.inf
        self.bins[-1] = np.inf
        s_binned = np.digitize(scores, self.bins) -1
        self.predictions = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            self.predictions[i] = (np.sum(y[s_binned == i]) + self.alpha)/ \
                                    (np.sum(s_binned == i) + self.alpha*2)

        return self

    def predict_proba(self, scores, *args, **kwargs):
        if len(np.shape(scores)) > 1:
            scores = scores[:,1]
        s_binned = np.digitize(scores, self.bins) - 1
        return self.predictions[s_binned]


l2_list = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
l2_odir = [1e3, 1e1, 1e-1, 1e-3, 1e-5, 1e-7]
#l2_list = [1e1, 1e0, 1e-1, 1e-2, 1e-3]
C_list = list(np.true_divide(1, l2_list, where=l2_list!=0))
n_bins = [5, 10, 15, 20, 25, 30]

MAP_CALIBRATORS = {
    'ovr_logistic_log': LogisticCalibration(C=C_list,
                                            log_transform=True,
                                            multi_class='ovr'),
    'ovr_logistic_logit': LogisticCalibration(C=C_list,
                                              log_transform=False,
                                              multi_class='ovr'),
    'logistic_log': LogisticCalibration(C=C_list,
                                        log_transform=True,
                                        multi_class='multinomial'),
    'logistic_logit': LogisticCalibration(C=C_list,
                                          log_transform=False,
                                          multi_class='multinomial'),
    'binning_width' :OneVsRestCalibrator(BinningCalibration(strategy='uniform',
                                                           n_bins=n_bins)),
    'binning_freq' :OneVsRestCalibrator(BinningCalibration(strategy='quantile',
                                                           n_bins=n_bins)),
    'binning_kmeans' :OneVsRestCalibrator(BinningCalibration(strategy='kmeans')), # Not working yet
    'uncalibrated': _DummyCalibration(),
    'isotonic': OneVsRestCalibrator(IsotonicCalibration(out_of_bounds='clip')),
    'sigmoid': OneVsRestCalibrator(SigmoidCalibration()),
    'beta': OneVsRestCalibrator(BetaCalibration(parameters="abm")),
    'beta_am': OneVsRestCalibrator(BetaCalibration(parameters="am")),
    'beta_ab': OneVsRestCalibrator(BetaCalibration(parameters="ab")),
    'ovr_dir_full': OneVsRestCalibrator(DirichletCalibrator(matrix_type='full')),
    'ovr_dir_full_l2': OneVsRestCalibrator(DirichletCalibrator(matrix_type='full',
                                             l2=l2_list)),
    'ovr_dir_diag': OneVsRestCalibrator(DirichletCalibrator(matrix_type='diagonal')),
    'ovr_dir_fixd': OneVsRestCalibrator(DirichletCalibrator(matrix_type='fixed_diagonal')),
    'dirichlet_keras': Dirichlet_NN(l2=10, mu=0.0001),
    'dirichlet_full': DirichletCalibrator(matrix_type='full'),
    'dirichlet_full_gen': GenerativeDirichletCalibrator(),
    'dirichlet_full_prefixdiag': DirichletCalibrator(matrix_type='full',
                                                     initializer='preFixDiag'),
    'dirichlet_full_comp_l2': DirichletCalibrator(matrix_type='full',
                                                  comp_l2=True,
                                             l2=l2_list),
    'dirichlet_full_prefixdiag_l2': DirichletCalibrator(matrix_type='full',
                                                        l2=l2_list,
                                                        initializer='preFixDiag'),
    'dirichlet_full_prefixdiag_comp_l2': DirichletCalibrator(matrix_type='full',
                                                             comp_l2=True,
                                                             l2=l2_list,
                                                             initializer='preFixDiag'),
    'dirichlet_full_l2_01': DirichletCalibrator(matrix_type='full', l2=0.1),
    'dirichlet_full_l2_001': DirichletCalibrator(matrix_type='full', l2=0.01),
    'dirichlet_full_l2_0001': DirichletCalibrator(matrix_type='full', l2=0.001),
    'dirichlet_full_l2_00001': DirichletCalibrator(matrix_type='full', l2=0.0001),
    'dirichlet_diag': DirichletCalibrator(matrix_type='diagonal'),
    #'dirichlet_fix_diag': DirichletCalibrator(matrix_type='fixed_diagonal'),
    'dirichlet_fix_diag': FixedDiagonalDirichletCalibrator(),
    #'dirichlet_full_t2': TypeIIDirichletCalibrator(),
    'temperature_scaling': TemperatureScaling(logit_constant=0.0),
    'vector_scaling': VectorScaling(logit_constant=0.0),
    #'matrix_scaling': MatrixScaling(reg_lambda_list=l2_list),
    #                                reg_mu_list=l2_list),
    'dirichlet_full_l2': FullDirichletCalibrator(reg_lambda_list=l2_list),
    #'dirichlet_full_l2': DirichletCalibrator(matrix_type='full',
    #                                         comp_l2=False,
    #                                         l2=l2_list),
    'dirichlet_odir_l2': FullDirichletCalibrator(reg_lambda_list=l2_odir,
                                                 reg_mu_list=l2_odir,
                                                 reg_norm=True)
#    'dirichlet_mixture': MixDir()
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

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):
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

        if X_val is not None:
            X_val, y_val = check_X_y(X_val, y_val, accept_sparse=['csc', 'csr', 'coo'],
                             multi_output=True)
            X_val, y_val = indexable(X_val, y_val)
            # TODO add scores of validation
            scores_val = self.base_estimator.predict_proba(X_val)
        else:
            scores_val = None

        self.calibrator = clone(MAP_CALIBRATORS[self.method])
        # TODO isotonic with binary y = (n_samples, ) fails, needs one-hot-enc.
        self.calibrator.fit(scores, y, X_val=scores_val, y_val=y_val, *args, **kwargs)
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
