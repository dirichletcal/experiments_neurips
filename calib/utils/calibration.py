from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import clone

from calib.models.calibration import _DummyCalibration
from calib.models.calibration import CalibratedModel
from functions import cross_entropy
from functions import brier_score
from functions import beta_test
from betacal import BetaCalibration
from calib.utils.functions import beta_test
from calib.utils.functions import fit_beta_moments
from calib.utils.functions import fit_beta_midpoint


def get_calibrated_scores(classifiers, methods, scores):
    probas = []
    for method in methods:
        p = np.zeros(len(scores))
        c_method = classifiers[method]
        for classifier in c_method:
            p += classifier.calibrator.predict(scores)
        probas.append(p / np.alen(c_method))
    return probas


def calibrate(classifier, x_cali, y_cali, method=None, score_type=None):
    ccv = CalibratedModel(base_estimator=classifier, method=method,
                          score_type=score_type)
    ccv.fit(x_cali, y_cali)
    return ccv


def cv_calibration(base_classifier, methods, x_train, y_train, x_test,
                   y_test, cv=3, score_type=None,
                   model_type='map-only', verbose=False):
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    mean_probas = {method: np.zeros(np.alen(y_test)) for method in methods}
    classifiers = {method: [] for method in methods}
    main_classifier = clone(base_classifier)
    rejected_count = 0
    if model_type == 'map-only':
        main_classifier.fit(x_train, y_train)
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            for method in methods:
                if verbose:
                    print("Calibrating with " + 'none' if method is None else
                          method)
                ccv = calibrate(classifier, x_c, y_c, method=method,
                                score_type=score_type)
                if method in ["beta_test_strict", "beta_test_relaxed"]:
                    test = beta_test(ccv.calibrator.calibrator_.map_,
                                     test_type="adev", scores=ccv._preproc(x_c))
                    if test["p-value"] < 0.05:
                        rejected_count += 1
                if model_type == 'map-only':
                    ccv.set_base_estimator(main_classifier,
                                           score_type=score_type)
                mean_probas[method] += ccv.predict_proba(x_test)[:, 1] / cv
                classifiers[method].append(ccv)
    if "beta_test_strict" in methods and rejected_count < cv:
        mean_probas["beta_test_strict"] = 0
        for classifier in classifiers["beta_test_strict"]:
            classifier.calibrator = _DummyCalibration()
            mean_probas["beta_test_strict"] += classifier.predict_proba(x_test)[:, 1] / cv
    if "beta_test_relaxed" in methods and rejected_count == 0:
        mean_probas["beta_test_relaxed"] = 0
        for classifier in classifiers["beta_test_relaxed"]:
            classifier.calibrator = _DummyCalibration()
            mean_probas["beta_test_relaxed"] += classifier.predict_proba(x_test)[:, 1] / cv
    losses = {method: cross_entropy(mean_probas[method], y_test) for method
              in methods}
    accs = {method: np.mean((mean_probas[method] >= 0.5) == y_test) for method
            in methods}
    briers = {method: brier_score(mean_probas[method], y_test) for method
              in methods}
    return accs, losses, briers, mean_probas, classifiers


def cv_calibration_map_differences(base_classifier, x_train, y_train, cv=3,
                                   score_type=None):
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    a = np.zeros((cv, 2))
    b = np.zeros((cv, 2))
    m = np.zeros((cv, 2))
    df_pos = None
    df_neg = None
    ccv = None
    for i, (train, cali) in enumerate(folds):
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
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    intervals = None
    for i, (train, cali) in enumerate(folds):
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
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    p_values = np.array([])
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores = ccv._preproc(x_test)
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
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    bins = np.linspace(0, 1, 101)
    improv_counts = np.zeros(100)
    total_counts = np.zeros(100)
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores_c = ccv._preproc(x_c)
            scores = ccv._preproc(x_test)
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
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    p_values = np.zeros(cv)
    improvements = np.zeros(cv)
    p_values_dist = np.zeros(cv)
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            ccv = calibrate(classifier, x_c, y_c, method="beta",
                            score_type=score_type)
            scores_c = ccv._preproc(x_c)
            scores = ccv._preproc(x_test)

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
