from __future__ import division
import time
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from numpy.testing import assert_array_equal

from calib.models.calibration import _DummyCalibration
from calib.models.calibration import CalibratedModel
from .functions import cross_entropy
from .functions import brier_score
from .functions import beta_test
from .functions import ECE, guo_ECE, classwise_ECE, full_ECE, pECE
from .functions import MCE
from betacal import BetaCalibration
from calib.utils.functions import beta_test
from calib.utils.functions import fit_beta_moments
from calib.utils.functions import fit_beta_midpoint

logger = logging.getLogger(__name__)

def get_calibrated_scores(classifiers, methods, scores):
    probas = []
    for method in methods:
        p = np.zeros(len(scores))
        c_method = classifiers[method]
        for classifier in c_method:
            p += classifier.calibrator.predict(scores)
        probas.append(p / np.alen(c_method))
    return probas


def cv_calibration(base_classifier, methods, x_train, y_train, x_test,
                   y_test, cv=3, score_type=None,
                    verbose=NameError, seed=None):
    ''' Train a classifier with the specified dataset and calibrate

    Parameters
    ----------
    base_classifier : string
        Name of the classifier to be trained and tested
    methods : list of strings
        List of all the calibrators to train and test
    x_train : array-like, shape (n_train_samples, n_features)
        Training data.
    y_train : array-like of integers, shape (n_train_samples,)
        Labels for each training sample in integer form
    x_test : array-like, shape (n_test_samples, n_features)
        Test data.
    y_test : array-like of integers, shape (n_test_samples,)
        Labels for each test sample in integer form
    cv : int
        Number of folds to perform in the training set, to train the classifier
        and the calibrator. The classifier is always trained in the bigger
        fold, while the calibrator is trained in the remaining 1 fold. This is
        repeated 'cv' times.
    score_type : string
        String indicating the function to call to obtain predicted
        probabilities from the classifier.
    verbose : ErrorType
    seed : int
        Seed for the stratified k folds
    Returns
    -------
    Each of the following dictionaries contain one entry with a calibrator and
    their corresponding information:

    accs : dict of float
        Mean accuracy for the inner folds
    losses : dict of float
        Mean Log-loss for the inner folds
    briers : dict of float
        Mean Brier score for the inner folds
    mean_probas : array of floats (n_samples_test, n_classes)
        Mean probability predictions for the inner folds and the test set
    classifiers : dict of list of classifier objects
        List of object classifiers
    mean_time : dict of float
        Mean calibration time for the inner folds
    '''
    # Ensure the same classes in train and test partitions
    assert_array_equal(np.unique(y_train), np.unique(y_test))

    # Prepare a binarizer
    binarizer = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    y_train_bin = binarizer.fit_transform(y_train)

    mean_probas = {method: np.zeros((y_test.shape[0], len(binarizer.classes_)))
                   for method in methods}
    classifiers = {method: [] for method in methods}
    exec_time = {method: [] for method in methods}
    train_acc = {method: 0 for method in methods}
    train_loss = {method: 0 for method in methods}
    train_brier = {method: 0 for method in methods}
    train_guo_ece = {method: 0 for method in methods}
    train_cla_ece = {method: 0 for method in methods}
    train_full_ece = {method: 0 for method in methods}
    train_mce = {method: 0 for method in methods}

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        print('Evaluation of split {} of {}'.format(i+1, cv))
        x_t = x_train[train]
        y_t = y_train[train]
        x_c = x_train[cali]
        y_c = y_train[cali]
        # Ensure the same classes in train and test partitions
        assert_array_equal(np.unique(y_t), np.unique(y_c))

        classifier = clone(base_classifier)
        classifier.fit(x_t, y_t)
        for method in methods:
            logger.debug("Calibrating with {}".format( 'none' if method is None
                                               else method))
            start = time.time()
            ccv = CalibratedModel(base_estimator=classifier, method=method,
                                  score_type=score_type)
            ccv.fit(x_c, y_c, X_val=x_t, y_val=y_t) # x_t and y_t for validation
            end = time.time()
            exec_time[method].append(end - start)
            predicted_proba = ccv.predict_proba(x_test)
            mean_probas[method] += predicted_proba / cv
            classifiers[method].append(ccv)

            predicted_proba = ccv.predict_proba(x_c)
            train_acc[method] += np.mean(predicted_proba.argmax(axis=1) == y_train[cali])/cv
            train_loss[method] += cross_entropy(predicted_proba, y_train_bin[cali])/cv
            train_brier[method] += brier_score(predicted_proba, y_train_bin[cali])/cv
            train_guo_ece[method] += guo_ECE(predicted_proba, y_train[cali])/cv
            train_cla_ece[method] += classwise_ECE(predicted_proba, y_train_bin[cali])/cv
            train_full_ece[method] += full_ECE(predicted_proba, y_train_bin[cali])/cv
            train_mce[method] += MCE(predicted_proba, y_train[cali])/cv

    y_test_bin = binarizer.transform(y_test)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
    # TODO we are doing a bootstrap of calibration methods... Shouldn't we
    # asses the performance of each individual calibrator and then compute the
    # mean? Is this the same?
    print('Computing Log-loss')
    losses = {method: cross_entropy(mean_probas[method], y_test_bin) for method
              in methods}
    print('Computing Accuracy')
    accs = {method: np.mean((mean_probas[method].argmax(axis=1)) == y_test) for method
            in methods}
    print('Computing Brier score')
    briers = {method: brier_score(mean_probas[method], y_test_bin) for method
              in methods}
    print('Computing confusion matrix')
    cms = {method: confusion_matrix(y_test, mean_probas[method].argmax(axis=1)) for method
              in methods}
    print('Computing binary guo_ECE')
    guo_eces = {method: guo_ECE(mean_probas[method], y_test) for method in methods}
    print('Computing classwise ECE')
    cla_eces = {method: classwise_ECE(mean_probas[method], y_test_bin) for method in methods}
    print('Computing full ECE')
    full_eces = {method: full_ECE(mean_probas[method], y_test_bin) for method in methods}
    print('Computing p-test binary Guo ECE')
    p_guo_eces = {method: pECE(mean_probas[method], y_test_bin, samples=1000,
                              ece_function=guo_ECE) for method in methods}
    print('Computing p-test classwise ECE')
    p_cla_eces = {method: pECE(mean_probas[method], y_test_bin, samples=1000,
                               ece_function=classwise_ECE) for method in methods}
    print('Computing p-test full ECE')
    p_full_eces = {method: pECE(mean_probas[method], y_test_bin, samples=1000) for method in methods}
    print('Computing MCE')
    mces = {method: MCE(mean_probas[method], y_test) for method in methods}
    mean_time = {method: np.mean(exec_time[method]) for method in methods}
    return (train_acc, train_loss, train_brier, train_guo_ece, train_cla_ece,
            train_full_ece, train_mce, accs, losses, briers, guo_eces,
            cla_eces, full_eces, p_guo_eces, p_cla_eces, p_full_eces, mces, cms,
            mean_probas, classifiers, mean_time)


def cv_calibration_map_differences(base_classifier, x_train, y_train, cv=3,
                                   score_type=None):
    a = np.zeros((cv, 2))
    b = np.zeros((cv, 2))
    m = np.zeros((cv, 2))
    df_pos = None
    df_neg = None
    ccv = None

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method='beta',
                            score_type=score_type)
            a[i] = ccv.a
            b[i] = ccv.b
            m[i] = ccv.m
            df_pos = ccv.df_pos
            df_neg = ccv.df_neg
    return a, b, m, df_pos, df_neg, ccv


def cv_confidence_intervals(base_classifier, x_train, y_train, x_test,
                            y_test, cv=2, score_type=None):
    intervals = None
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        if i == 0:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method=None,
                            score_type=score_type)

            scores = ccv.predict_proba(x_c)[:, 1]
            scores_test = ccv.predict_proba(x_test)[:, 1]
            ll_before = cross_entropy(scores_test, y_test)
            brier_before = brier_score(scores_test, y_test)

            calibrator = BetaCalibration(parameters="abm").fit(scores, y_c)

            ll_after = cross_entropy(calibrator.predict(scores_test), y_test)
            brier_after = brier_score(calibrator.predict(scores_test), y_test)

            original_map = calibrator.calibrator_.map_
            intervals = beta_test(original_map,
                                  test_type="adev", scores=scores)
            intervals["ll_diff"] = ll_after - ll_before
            intervals["bs_diff"] = brier_after - brier_before
    return intervals


def cv_p_improvement(base_classifier, x_train, y_train, x_test,
                            y_test, cv=2, score_type=None):
    p_values = np.array([])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores = ccv.base_estimator.predict_proba(x_test)
            scores_beta = ccv.predict_proba(x_test)[:, 1]

            ll_before = cross_entropy(scores, y_test)
            ll_after = cross_entropy(scores_beta, y_test)

            if ll_after < ll_before:
                test = beta_test(ccv.calibrator.calibrator_.map_,
                                 test_type="adev", scores=scores)
                p_values = np.append(p_values, test["p-value"])
    return p_values


def cv_p_improvement_correct(base_classifier, x_train, y_train, x_test,
                             y_test, cv=2, score_type=None):
    bins = np.linspace(0, 1, 101)
    improv_counts = np.zeros(100)
    total_counts = np.zeros(100)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores_c = ccv.base_estimator.predict_proba(x_c)
            scores = ccv.base_estimator.predict_proba(x_test)
            scores_beta = ccv.predict_proba(x_test)[:, 1]

            ll_before = cross_entropy(scores, y_test)
            ll_after = cross_entropy(scores_beta, y_test)

            test = beta_test(ccv.calibrator.calibrator_.map_,
                             test_type="adev", scores=scores_c)
            p_value = test["p-value"]
            idx = np.digitize(p_value, bins) - 1
            if idx == 100:
                idx = 99
            total_counts[idx] += 1
            if ll_after < ll_before:
                improv_counts[idx] += 1
    return {"improv": improv_counts, "total": total_counts}


def cv_all_p(base_classifier, x_train, y_train, x_test, y_test, cv=2,
             score_type=None):
    p_values = np.zeros(cv)
    improvements = np.zeros(cv)
    p_values_dist = np.zeros(cv)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train, cali) in enumerate(skf.split(X=x_train, y=y_train)):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores_c = ccv.base_estimator.predict_proba(x_c)
            scores = ccv.base_estimator.predict_proba(x_test)

            test = beta_test(ccv.calibrator.calibrator_.map_,
                             test_type="adev", scores=scores_c)
            p_values[i] = test["p-value"]

            scores_beta = ccv.predict_proba(x_test)[:, 1]

            ll_before = cross_entropy(scores, y_test)
            ll_after = cross_entropy(scores_beta, y_test)
            improvements[i] = ll_after < ll_before

            # df_pos = scores_c[y_c == 1]
            # df_neg = scores_c[y_c == 0]
            # alpha_pos, beta_pos = fit_beta_moments(df_pos)
            # alpha_neg, beta_neg = fit_beta_moments(df_neg)
            # a = alpha_pos - alpha_neg
            # if np.isnan(a):
            #     a = 0
            # if a > 100:
            #     a = 100
            # b = beta_neg - beta_pos
            # if np.isnan(b):
            #     b = 0
            # if b > 100:
            #     b = 100
            # prior_pos = len(df_pos) / len(scores_c)
            # prior_neg = len(df_neg) / len(scores_c)
            # m = fit_beta_midpoint(prior_pos, alpha_pos, beta_pos, prior_neg,
            #                       alpha_neg, beta_neg)
            # map = [a, b, m]
            # test = beta_test(map, test_type="adev", scores=scores_c)
            # p_values_dist[i] = test["p-value"]
    return p_values, improvements, p_values_dist
