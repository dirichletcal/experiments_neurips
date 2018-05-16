import os
import re
import pandas as pd
import numpy as np

from calib.utils.functions import to_latex

# Visualisations
from calib.utils.plots import df_to_heatmap

from scipy.stats import ranksums


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
    statistics = np.zeros_like(pvalues)
    for i, method_i in enumerate(table.columns.levels[1]):
        for j, method_j in enumerate(table.columns.levels[1]):
            sample_i = table[measure, method_i]
            sample_j = table[measure, method_j]
            statistic, pvalue = ranksums(sample_i, sample_j)
            pvalues[i, j] = pvalue
            statistics[i, j] = statistic
    index = pd.MultiIndex.from_product([table.columns.levels[1],
                                        ['statistic']])
    df_statistics = pd.DataFrame(statistics,
                                 index=table.columns.levels[1],
                                 columns=index)
    index = pd.MultiIndex.from_product([table.columns.levels[1],
                                        ['pvalue']])
    df_pvalues = pd.DataFrame(pvalues,
                              index=table.columns.levels[1],
                              columns=index)
    return df_statistics.join(df_pvalues)


def export_ranksums_to_latex(df_ranksums, filename, threshold=0.005,
                            caption='', label='', fontsize='\\tiny'):
    def pvalue_to_tex(s, p, threshold):
        s = '%.1f' % s
        if p < threshold:
            s = '\\bf{' + s + '}'
        return s

    statistics = df_ranksums.xs('statistic', axis=1, level=1, drop_level=False)
    pvalues = df_ranksums.xs('pvalue', axis=1, level=1, drop_level=False)

    table = np.empty((df_ranksums.index.shape[0],
                          df_ranksums.columns.levels[0].shape[0]),
                    dtype=np.object_)
    for i, method_i in enumerate(df_ranksums.index):
        for j, method_j in enumerate(df_ranksums.index):
            table[i, j] = pvalue_to_tex(statistics.iloc[i, j],
                                        pvalues.iloc[i, j],
                                        threshold)

    index = [x.replace('_', '\_') for x in df_ranksums.index]
    columns = [x.replace('_', '\_') for x in df_ranksums.columns.levels[0]]
    df = pd.DataFrame(table, index=index, columns=columns)

    tex_table = df.to_latex(escape=False)

    tex_table = ('\\begin{table}\n' + fontsize + '\n' + tex_table +
                '\\caption{{{}}}\n\\label{{{}}}\n'.format(caption, label) +
                '\\end{table}')
    with open(filename, 'w') as f:
        f.write(tex_table)


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
    df['method'] = df['method'].replace(to_replace='dirichlet', value='dir',
                                        regex=True)
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
        threshold = 0.005
        export_ranksums_to_latex(df_ranksums, filename, threshold=threshold,
                                 caption=('Wilcoxon rank-sum test statistic '
                                          'for every paired method for the '
                                          'measure of {}. Statistic is bold '
                                          'when p-value is smaller than '
                                          '{}').format(measure, threshold),
                                 label='tab:ranksum:{}'.format(measure)
                                )

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
                                          'applied on the classifier ' +
                                          '{} with the measure={}'
                                          ).format(classifier_name, measure),
                                 label='table:{}:{}'.format(classifier_name,
                                                            measure),
                                 add_std=False)
            file_basename = os.path.join(summary_path, classifier_name +
                                         '_dataset_vs_method_' + measure)
            with open(file_basename + '.tex', "w") as text_file:
                text_file.write(str_table)
            df_to_heatmap(table['mean'][measure], file_basename + '.svg',
                          title=measure)
            df_to_heatmap(table['mean'][measure], file_basename + '_rows.svg',
                          title='Normalised rows for ' + measure,
                          normalise_rows=True)


