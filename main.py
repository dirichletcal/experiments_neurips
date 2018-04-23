# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main.py # where -n is the number of workers (
# threads)
# Not parallelized (easier to debug):
#   python main.py

from __future__ import division
import argparse
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
import matplotlib.pyplot as plt

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

#methods = [None, 'beta', 'beta_ab', 'beta_am', 'isotonic', 'sigmoid', 'dirichlet_full']
methods = ['beta', 'beta_am', 'isotonic', 'sigmoid', 'dirichlet_full']
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

columns = ['dataset', 'method', 'mc', 'test_fold', 'acc', 'loss', 'brier',
           'c_probas']


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the experiments
                                     with the given arguments''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--classifier', dest='classifier_name', type=str,
                        default='nbayes',
                        help='''Classifier to use for evaluation''')
    parser.add_argument('-s', '--seed', dest='seed_num', type=int,
                        default=42,
                        help='Seed for the random number generator')
    parser.add_argument('-i', '--iterations', dest='mc_iterations', type=int,
                        default=10,
                        help='Number of Markov Chain iterations')
    parser.add_argument('-f', '--folds', dest='n_folds', type=int,
                        default=5,
                        help='Folds to create for cross-validation')
    parser.add_argument('-o', '--output_path', dest='results_path', type=str,
                        default='results_test',
                        help='''Path to store all the results''')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='''Show additional messages''')
    return parser.parse_args()


def df_to_heatmap(df, filename, title=None, figsize=(6,4), annotate=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    cax = ax.pcolor(df)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels([' '.join(row).strip() for row in df.index.values])
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels([' '.join(col).strip() for col in df.columns.values],
                       rotation = 45, ha="right")

    if annotate:
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.2f' % df.values[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
    fig.tight_layout()
    fig.savefig(filename)


def compute_all(args):
    (name, dataset, n_folds, mc, classifier_name, verbose) = args
    classifier = classifiers[classifier_name]
    score_type = score_types[classifier_name]
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
    # FIXME Change the binarization of the target to some other approach
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
                                                               model_type='full-stack',
                                                               verbose=verbose)

        for method in methods:
            m_text = 'None' if method is None else method
            df = df.append_rows([[name, m_text, mc, test_fold,
                                  accs[method], losses[method], briers[method],
                                  mean_probas[method]]])
    return df


def main(seed_num, mc_iterations, n_folds, classifier_name, results_path,
		 verbose):
    print(locals())
    results_path += '/' + classifier_name

    #dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
    #                         datasets_others))
    dataset_names = list(set(datasets_small_example))
    # dataset_names = datasets_big
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)

    data = Data(dataset_names=dataset_names)

    for name, dataset in data.datasets.items():
        df = MyDataFrame(columns=columns)
        print(dataset)

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs, [classifier_name], [verbose]]
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
    df_to_heatmap(table['mean'], os.path.join(results_path, 'main_results.svg'))

    # remove_list = [[], ['isotonic'], ['beta_am'], ['beta_ab'],
    #                ['beta', 'beta_ab'], ['beta_am', 'beta_ab'],
    #                [None, 'None', 'isotonic', 'sigmoid']]
    remove_list = [[]]
    for rem in remove_list:
        df_rem = df_all[np.logical_not(np.in1d(df_all.method, rem))]
        methods_rem = [method for method in methods if method not in rem]
        print(methods_rem)

        measures = (('acc', True), ('loss', False), ('brier', False))
        for measure, max_is_better in measures:
            print('-#-#-#-#-#-#-#-#-' + measure + '-#-#-#-#-#-#-#-#-#-#-#-#-#-')
            table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                       values=[measure], aggfunc=[np.mean, np.std])
            table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
            values = table.as_matrix()[:, :len(methods_rem)]
            #print(friedmanchisquare(*[values[:, x] for x in
            #                          np.arange(values.shape[1])]))
            print(p_value(values))
            table.to_csv(os.path.join(results_path,
                                      'main_{}{}.csv'.format(measure, methods_rem)))
            df_to_heatmap(table['mean'][measure],
                          os.path.join(results_path,
                                       '{}_dataset_vs_method.svg'.format(measure)),
                         title=measure)

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
