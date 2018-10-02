import os
import re
import pandas as pd
import numpy as np

from functools import partial

from calib.utils.functions import to_latex

# Visualisations
from calib.utils.plots import df_to_heatmap
from calib.utils.plots import export_critical_difference

from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata


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
    return paired_test(table, stats_func=ranksums)


def compute_mannwhitneyu(table):
    return paired_test(table, stats_func=partial(mannwhitneyu,
                                                 alternative='less'))


def compute_friedmanchisquare(table):
    '''
    Example:
        - n wine judges each rate k different wines. Are any of the k wines
        ranked consistently higher or lower than the others?
    Our Calibration case:
        - n datasets each rate k different calibration methods. Are any of the
        k calibration methods ranked consistently higher or lower than the
        others?
    This will output a statistic and a p-value
    SciPy does the following:
        - k: is the number of parameters passed to the function
        - n: is the lenght of each array passed to the function
    The two options for the given table are:
        - k is the datasets: table['mean'].values).tolist()
        - k is the calibration methods: table['mean'].T.values).tolist()
    '''
    return friedmanchisquare(*(table['mean'].T.values).tolist())


def paired_test(table, stats_func=ranksums):
    measure = table.columns.levels[0].values[0]
    pvalues = np.zeros((table.columns.shape[0], table.columns.shape[0]))
    statistics = np.zeros_like(pvalues)
    for i, method_i in enumerate(table.columns.levels[1]):
        for j, method_j in enumerate(table.columns.levels[1]):
            sample_i = table[measure, method_i]
            sample_j = table[measure, method_j]
            statistic, pvalue = stats_func(sample_i, sample_j)
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


def export_statistic_to_latex(df_statistic, filename, threshold=0.005,
                              caption='', label='', fontsize='\\small',
                              str_format='%.1f', position='!H'):
    def pvalue_to_tex(s, p, threshold):
        s = str_format % s
        if p < threshold:
            s = '\\bf{' + s + '}'
        return s

    statistics = df_statistic.xs('statistic', axis=1, level=1, drop_level=False)
    pvalues = df_statistic.xs('pvalue', axis=1, level=1, drop_level=False)

    table = np.empty((df_statistic.index.shape[0],
                          df_statistic.columns.levels[0].shape[0]),
                    dtype=np.object_)
    for i, method_i in enumerate(df_statistic.index):
        for j, method_j in enumerate(df_statistic.index):
            table[i, j] = pvalue_to_tex(statistics.iloc[i, j],
                                        pvalues.iloc[i, j],
                                        threshold)

    index = [x.replace('_', '\_') for x in df_statistic.index]
    columns = [x.replace('_', '\_') for x in df_statistic.columns.levels[0]]
    df = pd.DataFrame(table, index=index, columns=columns)

    tex_table = df.to_latex(escape=False)

    tex_table = ('\\begin{table}[' + position + ']\n\\centering\n' +
                 fontsize + '\n' + tex_table + ('\\caption{{{}}}\n' +
                 '\\label{{{}}}\n').format(caption, label) +
                 '\\end{table}')
    with open(filename, 'w') as f:
        f.write(tex_table)


def generate_summaries(df, summary_path):
    '''
    df:     pandas.DataFrame
        The dataframe needs at least the following columns
        - 'dataset': name of the dataset
        - 'n_classes':
        - 'n_features':
        - 'n_samples':
        - 'method': calibration method (or method to compare)
        - 'mc': Monte Carlo iteration
        - 'test_fold': Number of the test fold
        - 'train_acc': Training Accuracy
        - 'train_loss': Training log-loss
        - 'train_brier': Training Brier score
        - 'acc': Accuracy
        - 'loss': log-loss
        - 'brier': Brier score
        - 'exec_time': Mean execution time
        - 'classifier': Original classifier used to train

    '''
    df['method'] = df['method'].replace(to_replace='dirichlet', value='dir',
                                        regex=True)
    dataset_names = df['dataset'].unique()
    classifiers = df['classifier'].unique()

    # Export summary of datasets
    (df[['dataset', 'n_samples', 'n_features', 'n_classes']]
        .drop_duplicates()
        .set_index('dataset')
        .sort_index()
        .to_latex(os.path.join(summary_path, 'datasets.tex')))

    measures = (('acc', True), ('loss', False), ('brier', False),
                ('train_acc', True), ('train_loss', False),
                ('train_brier', False), ('exec_time', False))
    for measure, max_is_better in measures:
        print('# Measure = {}'.format(measure))
        table = df.pivot_table(index=['classifier'], columns=['method'],
                               values=[measure], aggfunc=[np.mean, np.std])

        str_table = to_latex(classifiers, table, precision=2,
                             table_size='small', max_is_better=max_is_better,
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

        # Export the Mean performance of each method
        table = df.pivot_table(index=['dataset', 'classifier', 'n_classes'],
                               columns=['method'],
                               values=[measure])
        table.columns = table.columns.droplevel()
        table.to_csv(os.path.join(summary_path, measure + '.csv'))

        # Print correlation results
        for method in ['pearson', 'kendall', 'spearman']:
            print('\n{} correlation for the measure {}'.format(method, measure))
            print(table.reset_index(level='n_classes').corr(method=method))

        table = df.pivot_table(index=['mc', 'test_fold', 'dataset',
                                      'classifier'], columns=['method'],
                               values=[measure])

        np.isfinite(table.values).mean(axis=0)

        # Wilcoxon rank-sum test two-tailed
        df_ranksums = compute_ranksums(table)

        filename = os.path.join(summary_path,
                                'ranksum_pvalues_{}.tex'.format(measure))
        threshold = 0.005
        export_statistic_to_latex(df_ranksums, filename, threshold=threshold,
                                 caption=('Wilcoxon rank-sum test statistic '
                                          'for every paired method for the '
                                          'measure of {}. Statistic is bold '
                                          'when p-value is smaller than '
                                          '{}').format(measure, threshold),
                                 label='tab:ranksum:{}'.format(measure)
                                )

        # Mann-Whitney rank test one-sided alternative is first is smaller than
        df_mannwhitneyu = compute_mannwhitneyu(table)

        filename = os.path.join(summary_path,
                                'mannwhitneyu_pvalues_{}.tex'.format(measure))
        export_statistic_to_latex(df_mannwhitneyu, filename, threshold=threshold,
                                  caption=('Mann-Whitney U test statistic '
                                           'one sided with alternative '
                                           'hypothesis the method in row i '
                                           'is less than the method in column j '
                                           'for every pair of methods for the '
                                           'measure of {}. Statistic is bold '
                                           'when the p-value is smaller than '
                                           '{}').format(measure, threshold),
                                  label='tab:mannwhitney:{}'.format(measure),
                                  str_format='%1.1e'
                                )

        ranking_table = np.zeros((len(classifiers),
                                  df.method.unique().shape[0]))
        num_datasets = np.zeros(len(classifiers), dtype='int')
        for i, classifier_name in enumerate(classifiers):
            print('- Classifier name = {}'.format(classifier_name))
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure],
                                               aggfunc=[np.mean, np.std])
            if max_is_better:
                table *= -1
            ranking_table[i] = table['mean'].apply(rankdata, axis=1).mean()
            num_datasets[i] = len(table)

            filename = os.path.join(summary_path, 'crit_diff_' +
                                    classifier_name + '_' +
                                    measure + '.pdf')

            export_critical_difference(avranks=ranking_table[i],
                                       num_datasets=table.shape[0],
                                       names=table.columns.levels[2],
                                       filename=filename)

        df_mean_rankings = pd.DataFrame(ranking_table, index=classifiers,
                                        columns=table.columns.levels[2])
        # TODO check that performing the ranking of the rankings is appropriate
        export_critical_difference(avranks=df_mean_rankings.apply(rankdata,
                                                                  axis=1).mean(),
                                   num_datasets=num_datasets.sum(),
                                   names=table.columns.levels[2],
                                   filename=os.path.join(summary_path,
                                                         'crit_diff_' +
                                                         measure + '.pdf'))
        str_table = to_latex(classifiers, df_mean_rankings, precision=1,
                             table_size='\\small',
                             max_is_better=False,
                             caption=('Ranking of calibration methods ' +
                                      'applied to each classifier ' +
                                      'with the measure={}'
                                      ).format(measure),
                             label='table:{}'.format(measure),
                             add_std=False,
                             column_names=df_mean_rankings.columns)
        file_basename = os.path.join(summary_path,
                                     '{}_rankings'.format(measure))
        with open(file_basename + '.tex', "w") as text_file:
            text_file.write(str_table)




        for classifier_name in classifiers:
            print('- Classifier name = {}'.format(classifier_name))
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure],
                                               aggfunc=[np.mean, np.std])

            # Perform a Friedman statistic test
            ftest = compute_friedmanchisquare(table)
            print(ftest)

            str_table = to_latex(dataset_names, table, precision=2,
                                 table_size='\\small',
                                 max_is_better=max_is_better,
                                 caption=('Ranking of calibration methods ' +
                                          'applied on the classifier ' +
                                          '{} with the measure={}' +
                                          '(Friedman statistic test ' +
                                          '= {:.2E}, p-value = {:.2E})'
                                          ).format(classifier_name, measure,
                                                  ftest.statistic, ftest.pvalue),
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


    for classifier_name in classifiers:
        table = df.pivot_table(values=['train_acc', 'train_loss',
                                       'train_brier', 'acc', 'loss', 'brier'],
                               index=['dataset', 'method'],
                               aggfunc=[np.mean, np.std])
        table.to_csv(os.path.join(summary_path, classifier_name + '_main_results.csv'))
        table.to_latex(os.path.join(summary_path, classifier_name + '_main_results.tex'))


def generate_summary_hist(df, summary_path):
    file_basename = os.path.join(summary_path, 'scores_histogram')
    df = df.sort_index()
    df_to_heatmap(df, file_basename + '.svg', title='Mean score histograms',
                  normalise_rows=True)

    for classifier in df.index.levels[1]:
        df_to_heatmap(df.loc[(slice(None), [classifier]),
                             :].reset_index(level='classifier', drop=True),
                      file_basename + '_' + classifier + '.svg',
                      title='Mean score histograms ' + classifier,
                      normalise_rows=True)
