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

# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.calibration import cv_calibration
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets
from calib.utils.functions import table_to_latex
from calib.utils.functions import to_latex
from calib.utils.functions import p_value

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import datasets_li2014
from data_wrappers.datasets import datasets_hempstalk2008
from data_wrappers.datasets import datasets_others
from data_wrappers.datasets import datasets_big
from data_wrappers.datasets import datasets_small_example

from utils.visualisations import df_to_heatmap

#methods = [None, 'beta', 'beta_ab', 'beta_am', 'isotonic', 'sigmoid', 'dirichlet_full']
methods = [None, 'multinomial', 'dirichlet_full']
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


def compute_all(args):
    (name, dataset, n_folds, mc, classifier_name, verbose) = args
    classifier = classifiers[classifier_name]
    score_type = score_types[classifier_name]
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True, random_state=mc)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
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


# FIXME seed_num is not being used at the moment
def main(seed_num, mc_iterations, n_folds, classifier_name, results_path,
		 verbose):
    global methods
    print(locals())
    results_path = os.path.join(results_path, classifier_name)

    #dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
    #                         datasets_others))
    dataset_names = list(set(datasets_small_example))
    #dataset_names = datasets_big
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)
    columns_hist = ['method', 'dataset'] + ['{}-{}'.format(i/10, (i+1)/10) for
                                            i in range(0,10)]
    df_all_hist = MyDataFrame(columns=columns_hist)

    data = Data(dataset_names=dataset_names)

    for name, dataset in data.datasets.items():
        df = MyDataFrame(columns=columns)
        print(dataset)
        if np.any(dataset.counts < n_folds):
            print(("At least one of the classes does not have enough samples "
                   "for {} folds").format(n_folds))
            # TODO Remove problematic class instead
            print("Removing dataset from experiments and skipping")
            dataset_names = [aux for aux in dataset_names if aux != name]
            continue

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs, [classifier_name], [verbose]]
        args = list(itertools.product(*args))

        dfs = futures.map(compute_all, args)

        df = df.concat(dfs)

        table = df[df.dataset == name].pivot_table(
                    values=['acc', 'loss', 'brier'],
                    index=['method'], aggfunc=[np.mean, np.std])

        print(table)
        print('Histogram of all the scores')
        hist_rows = []
        for method in methods:
            m_text = 'None' if method is None else method
            hist = np.histogram(
                            np.concatenate(
                                df[df.dataset == name][df.method ==
                                                       m_text]['c_probas'].values),
                                range=(0.0, 1.0))
            hist_rows.append([m_text, name] + list(hist[0]))
        df_all = df_all.append(df)
        df_all_hist = df_all_hist.append_rows(hist_rows)
        print(df_all_hist[df_all_hist['dataset'] == name])
        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
    table = df_all.pivot_table(values=['acc', 'loss', 'brier'],
                               index=['dataset', 'method'],
                               aggfunc=[np.mean, np.std])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df_all.to_csv(os.path.join(results_path, 'main_results_data_frame.csv'))
    df_all_hist = df_all_hist.set_index(['method', 'dataset'])
    df_all_hist.to_csv(os.path.join(results_path, 'score_histograms.csv'))
    df_all_hist.to_latex(os.path.join(results_path, 'score_histograms.tex'))

    table.to_csv(os.path.join(results_path, 'main_results.csv'))
    table.to_latex(os.path.join(results_path, 'main_results.tex'))

    # remove_list = [[], ['isotonic'], ['beta_am'], ['beta_ab'],
    #                ['beta', 'beta_ab'], ['beta_am', 'beta_ab'],
    #                [None, 'None', 'isotonic', 'sigmoid']]
    # Change None for 'None'
    methods = ['None' if method is None else method for method in methods]
    remove_list = [[]]
    for rem in remove_list:
        df_rem = df_all[np.logical_not(np.in1d(df_all.method, rem))]
        methods_rem = [method for method in methods if method not in rem]
        print(methods_rem)

        measures = (('acc', True), ('loss', False), ('brier', False))
        for measure, max_is_better in measures:
            print('-#-#-#-#-#-#-#-#-' + measure + '-#-#-#-#-#-#-#-#-#-#-#-#-#-')
            file_basename = os.path.join(results_path,
                                         'dataset_vs_method_' + measure)
            table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                       values=[measure], aggfunc=[np.mean, np.std])
            table_to_latex(dataset_names, methods_rem, table,
                                       max_is_better=max_is_better)
            str_table = to_latex(dataset_names, table,
                                       max_is_better=max_is_better)
            with open(file_basename + '.tex', "w") as text_file:
                text_file.write(str_table)
            values = table.as_matrix()[:, :len(methods_rem)]
            #print(friedmanchisquare(*[values[:, x] for x in
            #                          np.arange(values.shape[1])]))
            print('P-value = {}'.format(p_value(values)))
            table.to_csv(file_basename + '.csv')
            df_to_heatmap(table['mean'][measure], file_basename + '.svg',
                          title=measure)

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
