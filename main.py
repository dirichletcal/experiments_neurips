# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main.py # where -n is the number of workers (
# threads)
# Not parallelized (easier to debug):
#   python main.py

from __future__ import division
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import calib.models.adaboost as our
import sklearn.ensemble as their
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.calibration import cv_calibration
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets
from calib.utils.functions import table_to_latex
from calib.utils.functions import p_value

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import datasets_li2014
from data_wrappers.datasets import datasets_hempstalk2008
from data_wrappers.datasets import datasets_others
from data_wrappers.datasets import datasets_big
from data_wrappers.datasets import datasets_small_example

#methods = [None, 'beta', 'beta_ab', 'beta_am', 'isotonic', 'sigmoid']
methods = ['beta', 'beta_am', 'isotonic']
classifiers = {
                  'nbayes': GaussianNB(),
                  'logistic': LogisticRegression(),
                  'adao': our.AdaBoostClassifier(n_estimators=200),
                  'adas': their.AdaBoostClassifier(n_estimators=200),
                  'forest': RandomForestClassifier(n_estimators=200),
                  'mlp': MLPClassifier(),
                  'svm': SVC()
}
score_types = {
                  'nbayes': 'predict_proba',
                  'logistic': 'predict_proba',
                  'adao': 'predict_proba',
                  'adas': 'predict_proba',
                  'forest': 'predict_proba',
                  'mlp': 'predict_proba',
                  'svm': 'sigmoid'
}

seed_num = 42
mc_iterations = 10
n_folds = 5
classifier_name = 'nbayes'
results_path = 'results_test/' + classifier_name
classifier = classifiers[classifier_name]
score_type = score_types[classifier_name]

columns = ['dataset', 'method', 'mc', 'test_fold', 'acc', 'loss', 'brier',
           'c_probas']


def compute_all(args):
    (name, dataset, n_folds, mc) = args
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
    if np.alen(class_counts) > 2:
        majority = np.argmax(class_counts)
        t = np.zeros_like(dataset.target)
        t[dataset.target == majority] = 1
    else:
        t = dataset.target
    for test_fold in np.arange(n_folds):
        x_train, y_train, x_test, y_test = get_sets(dataset.data, t, test_fold,
                                                    test_folds)
        accs, losses, briers, mean_probas, cl = cv_calibration(classifier,
                                                               methods,
                                                               x_train, y_train,
                                                               x_test, y_test,
                                                               cv=3,
                                                               score_type=score_type,
                                                               model_type='full-stack')

        for method in methods:
            m_text = 'None' if method is None else method
            df = df.append_rows([[name, m_text, mc, test_fold,
                                  accs[method], losses[method], briers[method],
                                  mean_probas[method]]])
    return df


if __name__ == '__main__':
    #dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
    #                         datasets_others))
    dataset_names = list(set(datasets_small_example))
    # dataset_names = datasets_big
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)

    data = Data(dataset_names=dataset_names)

    for name, dataset in data.datasets.iteritems():
        df = MyDataFrame(columns=columns)
        print(dataset)

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs]
        args = list(itertools.product(*args))

        # if called with -m scoop
        if '__loader__' in globals():
            dfs = futures.map(compute_all, args)
        else:
            dfs = map(compute_all, args)

        df = df.concat(dfs)

        table = df[df.dataset == name].pivot_table(
                    values=['acc', 'loss', 'brier', 'c_probas'],
                    index=['method'], aggfunc=[np.mean, np.std])

        print(table)
        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        df_all = df_all.append(df)
    table = df_all.pivot_table(values=['acc', 'loss'], index=['dataset', 'method'],
                           aggfunc=[np.mean, np.std])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df_all.to_csv(os.path.join(results_path, 'main_results_data_frame.csv'))

    table.to_csv(os.path.join(results_path, 'main_results.csv'))
    table.to_latex(os.path.join(results_path, 'main_results.tex'))

    # remove_list = [[], ['isotonic'], ['beta_am'], ['beta_ab'],
    #                ['beta', 'beta_ab'], ['beta_am', 'beta_ab'],
    #                [None, 'None', 'isotonic', 'sigmoid']]
    remove_list = [[]]
    for rem in remove_list:
        df_rem = df_all[np.logical_not(np.in1d(df_all.method, rem))]
        methods_rem = [method for method in methods if method not in rem]
        print(methods_rem)
        print('-#-#-#-#-#-#-#-#-#-#-#-#-ACC-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['acc'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=True)
        accs = table.as_matrix()[:, :len(methods_rem)]
        # print friedmanchisquare(*[accs[:, x] for x in np.arange(accs.shape[1])])
        print(p_value(accs))
        table.to_csv(os.path.join(results_path, 'main_acc' + str(methods_rem) + '.csv'))

        print('-#-#-#-#-#-#-#-#-#-#-#-LOSS-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['loss'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
        losses = table.as_matrix()[:, :len(methods_rem)]
        # print friedmanchisquare(*[losses[:, x] for x in np.arange(losses.shape[1])])
        print(p_value(losses))
        table.to_csv(os.path.join(results_path, 'main_loss' + str(methods_rem) + '.csv'))

        print('-#-#-#-#-#-#-#-#-#-#-#-BRIER-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['brier'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
        briers = table.as_matrix()[:, :len(methods_rem)]
        # print friedmanchisquare(*[briers[:, x] for x in np.arange(briers.shape[1])])
        print(p_value(briers))
        table.to_csv(os.path.join(results_path, 'main_brier' + str(methods_rem) + '.csv'))

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
