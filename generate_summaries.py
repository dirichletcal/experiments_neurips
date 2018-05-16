#!/usr/bin/env python
import os
import re
from argparse import ArgumentParser
import pandas as pd
import numpy as np

from calib.utils.functions import to_latex

from scipy.stats import ranksums


def parse_arguments():
    parser = ArgumentParser(description=("Generates a summary of all the " +
                                         "experiments in the subfolders of " +
                                         "the specified path"))
    parser.add_argument("results_path", metavar='PATH', type=str,
                        default='results',
                        help="Path with the result folders to summarize.")
    parser.add_argument("summary_path", metavar='SUMMARY', type=str,
                        default=None, nargs='?',
                        help="Path to store the summary.")
    return parser.parse_args()


def load_all_csv(results_path, expression=".*.csv"):
    regexp = re.compile(expression)
    filename_list = []
    df_list = []
    for root, subdirs, files in os.walk(results_path, followlinks=True):
        file_list = list(filter(regexp.match, files))
        for filename in file_list:
            if filename in filename_list:
                continue
            filename_list += filename
            classifier = filename.split('_')[0]
            df_list.append(pd.read_csv(os.path.join(root, filename)))
            df_list[-1]['classifier'] = classifier
            df_list[-1]['filename'] = filename

    df = pd.concat(df_list)
    return df


def create_summary_path(summary_path, results_path='./'):
    if summary_path is None:
        summary_path = os.path.join(results_path, 'summary')

    # Creates summary path if it does not exist
    if not os.path.exists(summary_path):
        print(summary_path)
        os.makedirs(summary_path)
    return summary_path


def compute_ranksums(table):
    measure = table.columns.levels[0].values[0]
    pvalues = np.zeros((table.columns.shape[0], table.columns.shape[0]))
    for i, method_i in enumerate(table.columns.levels[1]):
        for j, method_j in enumerate(table.columns.levels[1]):
            sample_i = table[measure, method_i]
            sample_j = table[measure, method_j]
            statistic, pvalue = ranksums(sample_i, sample_j)
            pvalues[i, j] = pvalue
    df_ranksums = pd.DataFrame(pvalues, index=table.columns.levels[1],
                               columns=table.columns.levels[1])
    return df_ranksums


def export_ranksums_to_latex(df_ranksums, filename):
    def pvalue_to_tex(x):
        string = '%.1e' % x
        if x < 0.04:
            string = '\\bf{' + string + '}'
        return string

    df_ranksums = df_ranksums.applymap(pvalue_to_tex)
    df_ranksums.index = [x.replace('_', '\_') for x in df_ranksums.index]
    df_ranksums.columns = [x.replace('_', '\_') for x in
                           df_ranksums.columns]
    df_ranksums.to_latex(filename, escape=False)


def generate_summaries(df, summary_path):
    '''
    df:     pandas.DataFrame
        The dataframe needs at least the following columns
        - 'dataset': name of the dataset
        - 'method': calibration method (or method to compare)
        - 'mc': Monte Carlo iteration
        - 'test_fold': Number of the test fold
        - 'acc': Accuracy
        - 'loss': A loss
        - 'brier': Brier score
        - 'classifier': Original classifier used to train

    '''
    dataset_names = df['dataset'].unique()
    classifiers = df['classifier'].unique()
    measures = (('acc', True), ('loss', False), ('brier', False))
    for measure, max_is_better in measures:
        table = df.pivot_table(index=['classifier'], columns=['method'],
                               values=[measure], aggfunc=[np.mean, np.std])

        str_table = to_latex(classifiers, table, precision=2,
                             table_size='tiny', max_is_better=max_is_better,
                             caption=('Ranking of calibration methods ' +
                                      'applied on different classifiers ' +
                                      'with the mean measure={}'
                                      ).format(measure),
                             label='table:mean:{}'.format(measure))

        table = df.pivot_table(index=['mc', 'test_fold', 'dataset',
                                      'classifier'], columns=['method'],
                               values=[measure], aggfunc=[len])

        try:
            assert(np.alltrue(table.values == 1))
        except AssertionError as e:
            print(e)

        table = df.pivot_table(index=['mc', 'test_fold', 'dataset',
                                      'classifier'], columns=['method'],
                               values=[measure])

        np.isfinite(table.values).mean(axis=0)

        df_ranksums = compute_ranksums(table)

        filename = os.path.join(summary_path,
                                'ranksum_pvalues_{}.tex'.format(measure))
        export_ranksums_to_latex(df_ranksums, filename)

        for classifier_name in classifiers:
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure],
                                               aggfunc=[np.mean, np.std])

            str_table = to_latex(dataset_names, table, precision=2,
                                 table_size='\\tiny',
                                 max_is_better=max_is_better,
                                 caption=('Ranking of calibration methods ' +
                                          'applied on the classifier' +
                                          '{} with the measure={}'
                                          ).format(classifier_name, measure),
                                 label='table:{}:{}'.format(classifier_name,
                                                            measure),
                                 add_std=False)
            file_basename = os.path.join(summary_path, classifier_name +
                                         '_dataset_vs_method_' + measure)
            with open(file_basename + '.tex', "w") as text_file:
                text_file.write(str_table)


def main(results_path, summary_path):
    df = load_all_csv(results_path, ".*raw_results.csv")
    summary_path = create_summary_path(summary_path, results_path)
    generate_summaries(df, summary_path)


if __name__ == '__main__':
    # __test_1()
    args = parse_arguments()
    main(**vars(args))
