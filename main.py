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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import calib.models.adaboost as our
from calib.models.classifiers import MockClassifier
import sklearn.ensemble as their
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Parallelization
import itertools
import scoop
from scoop import futures, shared, logger

# Our classes and modules
from calib.utils.calibration import cv_calibration
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets
from calib.utils.functions import table_to_latex
from calib.utils.functions import to_latex
from calib.utils.functions import p_value

from calib.utils.summaries import create_summary_path
from calib.utils.summaries import generate_summaries
from calib.utils.summaries import generate_summary_hist

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import datasets_non_binary


classifiers = {
      'mock': MockClassifier(),
      'nbayes': GaussianNB(),
      'logistic': LogisticRegression(),
      'adao': our.AdaBoostClassifier(n_estimators=200),
      'adas': their.AdaBoostClassifier(n_estimators=200),
      'forest': RandomForestClassifier(n_estimators=200),
      'mlp': MLPClassifier(),
      'svm': SVC(probability=True)
}
score_types = {
      'mock': 'predict_proba',
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

save_columns = ['dataset', 'method', 'mc', 'test_fold', 'acc', 'loss', 'brier']


def comma_separated_strings(s):
    try:
        return s.split(',')
    except ValueError:
        msg = "Not a valid comma separated list: {}".format(s)
        raise argparse.ArgumentTypeError(msg)


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
    parser.add_argument('--inner-folds', dest='inner_folds', type=int,
                        default=3,
                        help='''Folds to perform in any given training fold to
                                train the different calibration methods''')
    parser.add_argument('-o', '--output-path', dest='results_path', type=str,
                        default='results_test',
                        help='''Path to store all the results''')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true', default=False,
                        help='''Show additional messages''')
    parser.add_argument('-d', '--datasets', dest='datasets',
                        type=comma_separated_strings,
                        default=['iris', 'car'],
                        help='''Comma separated dataset names or one of the
                        defined groups in the datasets package''')
    parser.add_argument('-m', '--methods', dest='methods',
                        type=comma_separated_strings,
                        default=['None', 'beta', 'beta_am', 'isotonic',
                                 'sigmoid', 'dirichlet_full', 'dirichlet_diag',
                                 'dirichlet_fix_diag'],
                        help='''Comma separated calibration methods''')
    return parser.parse_args()


def compute_all(args):
    (name, n_folds, inner_folds, mc, classifier_name, methods, verbose) = args
    dataset = shared.getConst(name)
    classifier = classifiers[classifier_name]
    score_type = score_types[classifier_name]
    logger.info(locals())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=mc)
    df = MyDataFrame(columns=columns)
    class_counts = np.bincount(dataset.target)
    t = dataset.target
    fold_id = 0
    for train_idx, test_idx in skf.split(X=dataset.data, y=dataset.target):
        x_train, y_train = dataset.data[train_idx], dataset.target[train_idx]
        x_test, y_test = dataset.data[test_idx], dataset.target[test_idx]
        results = cv_calibration(classifier, methods, x_train, y_train, x_test,
                                 y_test, cv=inner_folds, score_type=score_type,
                                 model_type='full-stack', verbose=verbose,
                                 seed=mc)
        accs, losses, briers, mean_probas, cl = results

        for method in methods:
            df = df.append_rows([[name, method, mc, fold_id,
                                  accs[method], losses[method], briers[method],
                                  mean_probas[method]]])
        fold_id += 1
    return df


# FIXME seed_num is not being used at the moment
def main(seed_num, mc_iterations, n_folds, classifier_name, results_path,
		 verbose, datasets, inner_folds, methods):
    logger.info(locals())
    results_path = os.path.join(results_path, classifier_name)

    dataset_names = datasets
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)
    columns_hist = ['method', 'dataset'] + ['{}-{}'.format(i/10, (i+1)/10) for
                                            i in range(0,10)]
    df_all_hist = MyDataFrame(columns=columns_hist)

    data = Data(dataset_names=dataset_names)

    for name, dataset in data.datasets.items():
        df = MyDataFrame(columns=columns)
        logger.info(dataset)
        # Assert that every class has enough samples to perform the two
        # cross-validataion steps (classifier + calibrator)
        smaller_count = min(dataset.counts)
        if (smaller_count < n_folds) or \
           ((smaller_count*(n_folds-1)/n_folds) < inner_folds):
            logger.warn(("At least one of the classes does not have enough "
                         "samples for outer {} folds and inner {} folds"
                        ).format(n_folds, inner_folds))
            # TODO Remove problematic class instead
            logger.warn("Removing dataset from experiments and skipping")
            dataset_names = [aux for aux in dataset_names if aux != name]
            continue

        mcs = np.arange(mc_iterations)
        logger.info(dataset)
        shared.setConst(**{name: dataset})
        # All the arguments as a list of lists
        args = [[name], [n_folds], [inner_folds], mcs, [classifier_name],
                [methods], [verbose]]
        args = list(itertools.product(*args))

        # If only one worker, then do not use scoop
        if scoop.SIZE != 1:
            logger.info('Jobs will be deployed in {} workers'.format(scoop.SIZE))
            map_f = futures.map
        else:
            map_f = map

        dfs = map_f(compute_all, args)

        df = df.concat(dfs)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        df[save_columns].to_csv(os.path.join(results_path, classifier_name +
                                             '_' + name + '_raw_results.csv'))

        table = df[df.dataset == name].pivot_table(
                    values=['acc', 'loss', 'brier'],
                    index=['method'], aggfunc=[np.mean, np.std])

        logger.info(table)
        logger.info('Histogram of all the scores')
        hist_rows = []
        for method in methods:
            hist = np.histogram(
                            np.concatenate(
                                df[df.dataset == name][df.method ==
                                                       method]['c_probas'].values),
                                range=(0.0, 1.0))
            hist_rows.append([method, name] + list(hist[0]))
        df_all = df_all.append(df)
        df_all_hist = df_all_hist.append_rows(hist_rows)
        logger.info(df_all_hist[df_all_hist['dataset'] == name])
        logger.info('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')

    table = df_all.pivot_table(values=['acc', 'loss', 'brier'],
                               index=['dataset', 'method'],
                               aggfunc=[np.mean, np.std])

    df_all.to_csv(os.path.join(results_path, classifier_name + '_main_results_data_frame.csv'))
    df_all_hist = df_all_hist.set_index(['method', 'dataset'])
    df_all_hist.to_csv(os.path.join(results_path, classifier_name + '_score_histograms.csv'))
    df_all_hist.to_latex(os.path.join(results_path, classifier_name + '_score_histograms.tex'))

    table.to_csv(os.path.join(results_path, classifier_name + '_main_results.csv'))
    table.to_latex(os.path.join(results_path, classifier_name + '_main_results.tex'))

    df_all['classifier'] = classifier_name
    generate_summaries(df_all, results_path)

    generate_summary_hist(df_all_hist.astype(float), results_path)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))






