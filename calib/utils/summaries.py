import os
import re
import pandas as pd
import numpy as np

from functools import partial

from calib.utils.functions import to_latex

# Visualisations
from calib.utils.plots import df_to_heatmap
from calib.utils.plots import export_dataset_analysis
from calib.utils.plots import export_critical_difference

from scipy.stats import ranksums
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata

import matplotlib.pyplot as pyplot

pd.set_option('display.width', 1000)


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
            try:
                df_list.append(pd.read_csv(os.path.join(root, filename)))
                df_list[-1]['classifier'] = classifier
                df_list[-1]['filename'] = filename
            except pd.errors.EmptyDataError as e:
                print(e)
                print('Classifier = {}, filename = {}'.format(classifier,
                    filename))

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
    return friedmanchisquare(*(table.T.values).tolist())


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


def summarise_confusion_matrices(df, summary_path):
    def MakeList(x):
        T = tuple(x)
        if len(T) > 1:
            return T
        else:
            return T[0]
    def confusion_matrix(string):
        cm = np.fromstring(''.join(c for c in string if c in '0123456789 '), sep=' ')
        cm = cm.reshape(int(np.sqrt(len(cm))), -1)
        return cm
    df['confusion_matrix'] = df['confusion_matrix'].apply(confusion_matrix)
    for calibrator in df['method'].unique():
        df_aux = df[df['method'] == calibrator]
        df_aux = df_aux.pivot_table(index=['dataset'], columns=['classifier'],
                                    values=['confusion_matrix'], aggfunc=MakeList)
        fig = pyplot.figure(figsize=(df_aux.shape[1]*3, df_aux.shape[0]*3))
        fig.suptitle(calibrator)
        ij = 1
        for i, dat in enumerate(df_aux.index):
            for j, cla in enumerate(df_aux.columns.levels[1]):
                values = df_aux.loc[dat, ('confusion_matrix', cla)]
                ax = fig.add_subplot(len(df_aux), len(df_aux.columns), ij)
                if j == 0:
                    ax.set_ylabel(dat)
                if i == 0:
                    ax.set_title(cla)
                ij += 1
                if values is None:
                    continue
                cms = np.stack(values).mean(axis=0)
                # FIXME solve problem here, it seems that values is always
                # empty?
                if isinstance(cms, np.float):
                    continue
                cax = ax.pcolor(cms)
                middle_value = (cms.max() + cms.min())/2.0
                fontsize = min((30/(cms.shape[0]-2), 9))
                for y in range(cms.shape[0]):
                    for x in range(cms.shape[1]):
                        color = 'white' if middle_value > cms[y, x] else 'black'
                        ax.text(x + 0.5, y + 0.5, '%.1f' % cms[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color=color, fontsize=fontsize
                                 )
                ax.invert_yaxis()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(summary_path,
            'confusion_matrices_{}.svg'.format(calibrator)))


def summarise_hyperparameters(df, summary_path):
    def MakeList(x):
        T = tuple(x)
        if len(T) > 1:
            return T
        else:
            return T[0]

    # Histograms of parameters
    MAP_METHOD = {'binning_freq': 'n_bins=(?P<bins>\w+), ',
                  'binning_width': 'n_bins=(?P<bins>\w+), ',
                  'dir_full_l2': " 'l2': ([0-9\.\-e]+),",
                  'dir_full_comp_l2': " 'l2': ([0-9\.\-e]+),",
                  'dirichlet_full_prefixdiag_l2': " 'l2': ([0-9\.\-e]+),",
                  'mlr_log': " 'C': ([0-9\.\-e]+),",
                  'mlr_logit': " 'C': ([0-9\.\-e]+),",
                  'ovr_mlr_log': " 'C': ([0-9\.\-e]+),",
                  'ovr_mlr_logit': " 'C': ([0-9\.\-e]+),",
                 }
    for key, regex in MAP_METHOD.items():
        df_aux = df[df['method'] == key][['dataset', 'classifier', 'calibrators']]
        if len(df_aux) == 0:
            continue
        df_aux['calibrators'] = df_aux['calibrators'].apply(lambda x:
                                                            np.array(re.findall(regex,
                                                                                x)).astype(float))
        df_aux = df_aux.pivot_table(index=['dataset'], columns=['classifier'],
                                    values=['calibrators'], aggfunc=MakeList)
        all_unique = df_aux.values.flatten()
        all_unique = all_unique[all_unique != None]
        all_unique = np.unique(np.hstack(all_unique).flatten())
        fig = pyplot.figure(figsize=(df_aux.shape[1]*3, df_aux.shape[0]*3))
        fig.suptitle(key)
        ij = 1
        for i, dat in enumerate(df_aux.index):
            for j, cla in enumerate(df_aux.columns.levels[1]):
                values = df_aux.loc[dat, ('calibrators', cla)]
                ax = fig.add_subplot(len(df_aux), len(df_aux.columns), ij)
                if j == 0:
                    ax.set_ylabel(dat)
                if i == 0:
                    ax.set_title(cla)
                ij += 1
                if values is None:
                    continue
                parameters = np.concatenate(values).flatten()
                uniq, counts = np.unique(parameters, return_counts=True)
                missing_uniq = []
                missing_counts = []
                for all_u in all_unique:
                    if all_u not in uniq:
                        missing_uniq.append(all_u)
                        missing_counts.append(0)
                uniq = np.concatenate((uniq, missing_uniq))
                counts = np.concatenate((counts, missing_counts))
                sorted_idx = np.argsort(uniq)
                uniq = [np.format_float_scientific(x, precision=2) for x in uniq[sorted_idx]]
                counts = counts[sorted_idx]
                ax.bar(uniq, counts)
                ax.set_xticklabels(uniq, rotation=45, ha='right')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(summary_path, 'bars_hyperparameters_{}.svg'.format(key)))

    # heatmaps of parameters
    # FIXME change MAP method as it is not being used
    def weight_matrix(string):
        solution = re.findall("'weights_': array(.*?)]]\)", string, flags=re.DOTALL)
        matrices = []
        for s in solution:
            x = np.fromstring(''.join(c for c in s if c in
                                                  '0123456789.-e+,'), sep=',')
            x = x.reshape(int(np.floor(np.sqrt(len(x)))), -1)
            col_sums = np.sum(x,axis=0)
            amount_to_shift = ( col_sums[:-1] - np.diag(x) ) / (x.shape[0]-1)
            x = x - np.concatenate((amount_to_shift,[0]))
            x[:,-1] = x[:,-1] - col_sums[-1] / x.shape[0]
            matrices.append(x)
        return matrices

    def weights_keras(string):
        coeficients = re.findall("'weights': \[array(.*?)]]", string, flags=re.DOTALL)
        intercepts = re.findall(", array\(\[(.*?)]", string, flags=re.DOTALL)
        matrices = []
        for coef, inter in zip(coeficients, intercepts):
            coef = np.fromstring(''.join(c for c in coef if c in
                                                  '0123456789.-e+,'), sep=',')
            coef = coef.reshape(int(np.floor(np.sqrt(len(coef)))), -1)

            inter = np.fromstring(''.join(c for c in inter if c in
                                                  '0123456789.-e+,'), sep=',')
            x = np.vstack((coef.T, inter)).T
            matrices.append(x)
        return matrices

    def coef_intercept_matrix(string):
        coeficients = re.findall("'coef_': array(.*?)]]\)", string, flags=re.DOTALL)
        intercepts = re.findall("'intercept_': array(.*?)]\)", string, flags=re.DOTALL)
        matrices = []
        for coef, inter in zip(coeficients, intercepts):
            coef = np.fromstring(''.join(c for c in coef if c in
                                                  '0123456789.-e+,'), sep=',')
            coef = coef.reshape(int(np.floor(np.sqrt(len(coef)))), -1)

            inter = np.fromstring(''.join(c for c in inter if c in
                                                  '0123456789.-e+,'), sep=',')
            x = np.vstack((coef.T, inter)).T
            matrices.append(x)
        return matrices

    MAP_METHOD = {'dir_full_l2': weight_matrix,
                  'dir_keras': weights_keras,
                  'dir_full_gen': weight_matrix,
                  'dir_full_comp_l2': weight_matrix,
                  'dirichlet_full_prefixdiag_l2': weight_matrix,
                  'logistic_log': coef_intercept_matrix,
                  'logistic_logit': coef_intercept_matrix
                 }

    for key, function in MAP_METHOD.items():
        df_aux = df[df['method'] == key][['dataset', 'classifier', 'calibrators']]
        if len(df_aux) == 0:
            continue
        df_aux['calibrators'] = df_aux['calibrators'].apply(function)
        df_aux = df_aux.pivot_table(index=['dataset'], columns=['classifier'],
                                    values=['calibrators'],
                                    aggfunc=MakeList)
        fig = pyplot.figure(figsize=(df_aux.shape[1]*3, df_aux.shape[0]*3))
        fig.suptitle(key)
        ij = 1
        for i, dat in enumerate(df_aux.index):
            for j, cla in enumerate(df_aux.columns.levels[1]):
                values = df_aux.loc[dat, ('calibrators', cla)]
                ax = fig.add_subplot(len(df_aux), len(df_aux.columns), ij)
                if j == 0:
                    ax.set_ylabel(dat)
                if i == 0:
                    ax.set_title(cla)
                ij += 1
                if values is None:
                    continue
                # Stacking (#iter x #crossval x #crossval) on first dimension
                parameters = np.concatenate(values).mean(axis=0)
                # FIXME solve problem here, it seems that values is always
                # empty?
                if isinstance(parameters, np.float):
                    continue
                cax = ax.pcolor(parameters)
                middle_value = (parameters.max() + parameters.min())/2.0
                fontsize = min((20/(parameters.shape[0]-2), 9))
                for y in range(parameters.shape[0]):
                    for x in range(parameters.shape[1]):
                        color = 'white' if middle_value > parameters[y, x] else 'black'
                        ax.text(x + 0.5, y + 0.5, '%.e' % parameters[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color=color, fontsize=fontsize
                                 )
                ax.invert_yaxis()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(summary_path, 'heatmap_weights_{}.svg'.format(key)))


def generate_summaries(df, summary_path, table_size='small',
        hyperparameters=True, confusion_matrices=True):
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
        - 'calibrators': List of calibrators with their parameters

    '''
    # Shorten some names
    df['method'] = df['method'].replace(to_replace='dirichlet', value='dir',
                                        regex=True)
    df['method'] = df['method'].replace(to_replace='binning', value='bin',
                                        regex=True)
    df['method'] = df['method'].replace(to_replace='logistic', value='mlr',
                                        regex=True)
    df['method'] = df['method'].replace(to_replace='uncalibrated', value='uncal',
                                        regex=True)
    dataset_names = df['dataset'].unique()
    classifiers = df['classifier'].unique()

    # Assert that all experiments have finished
    for column in ['method', 'classifier']:
        for measure in ['acc', 'loss', 'brier']:
            df_count = df.pivot_table(index=['dataset'], columns=[column],
                                      values=[measure], aggfunc='count')
            file_basename = os.path.join(summary_path,
                                         'results_count_{}_{}'.format(column,
                                                                      measure))
            df_to_heatmap(df_count, file_basename + '.svg',
                          title='Results count finite ' + measure, cmap='Greys_r')

    # Export summary of datasets
    (df[['dataset', 'n_samples', 'n_features', 'n_classes']]
        .drop_duplicates()
        .set_index('dataset')
        .sort_index()
        .to_latex(os.path.join(summary_path, 'datasets.tex')))

    if hyperparameters:
        print('Generating summary of hyperparameters')
        summarise_hyperparameters(df, summary_path)

    if confusion_matrices:
        print('Generating summary of confusion matrices')
        summarise_confusion_matrices(df, summary_path)

    measures = (('acc', True), ('loss', False), ('brier', False),
                ('train_acc', True), ('train_loss', False),
                ('train_brier', False), ('exec_time', False))
    for measure, max_is_better in measures:
        print('# Measure = {}'.format(measure))
        if 'train_' not in measure:
            filename = os.path.join(summary_path,
                                'n_samples_scatter')
            export_dataset_analysis(df, measure, filename)

        table = df.pivot_table(index=['classifier'], columns=['method'],
                               values=[measure], aggfunc=[np.mean, np.std])

        str_table = to_latex(classifiers, table, precision=2,
                             table_size=table_size, max_is_better=max_is_better,
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

        if 'train_' not in measure:
            # Export the Mean performance of each method
            table = df.pivot_table(index=['dataset', 'classifier', 'n_classes',
                                          'n_samples'],
                                   columns=['method'],
                                   values=[measure])
            table.columns = table.columns.droplevel()
            table.to_csv(os.path.join(summary_path, measure + '.csv'))

            # Print correlation results
            method = 'spearman' # for non-linear ranking correlations
            #method = 'pearson' # for linear ranking correlations
            print('\n{} correlation for the measure {}'.format(method, measure))
            corr_test = table.reset_index(level=['n_classes', 'n_samples']).corr(method=method)
            print(corr_test)

            if ('train_' + measure) in [m[0] for m in measures]:
                table = df.pivot_table(index=['dataset', 'classifier', 'n_classes',
                                              'n_samples'],
                                       columns=['method'],
                                       values=['train_' + measure])
                table.columns = table.columns.droplevel()
                table.to_csv(os.path.join(summary_path, 'train_' + measure + '.csv'))
                print('\n{} correlation for the measure {}'.format(method, 'train_' + measure))
                corr_train = table.reset_index(level=['n_classes', 'n_samples']).corr(method=method)
                print(corr_train)
                print('\n{} correlation difference of test - training for the measure {}'.format(method, measure))
                print(corr_test - corr_train)


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

            # Perform a Friedman statistic test
            # Remove datasets in which one of the experiments failed
            table = table[~table.isna().any(axis=1)]
            ftest = compute_friedmanchisquare(table['mean'])
            print(ftest)

            str_table = to_latex(dataset_names, table, precision=2,
                                 table_size=table_size,
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

        #for i, classifier_name in enumerate(classifiers):
            print('- Classifier name = {}'.format(classifier_name))
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure],
                                               aggfunc=[np.mean, np.std])
            # Remove datasets in which one of the experiments failed
            table = table[~table.isna().any(axis=1)]
            if max_is_better:
                table *= -1
            ranking_table[i] = table['mean'].apply(rankdata, axis=1).mean()
            num_datasets[i] = len(table)

            filename = os.path.join(summary_path, 'crit_diff_' +
                                    classifier_name + '_' +
                                    measure + '.pdf')

            print(('Critical Difference computed with avranks of shape {} ' +
                   'for {} datasets').format(np.shape(ranking_table[i]),
                                         table.shape[0]))
            export_critical_difference(avranks=ranking_table[i],
                                       num_datasets=table.shape[0],
                                       names=table.columns.levels[2],
                                       filename=filename,
                                       title='(p-value = {:.2e}, #D = {})'.format(ftest.pvalue, table.shape[0]))

        ## Export the summary of all rankings
        df_mean_rankings = pd.DataFrame(ranking_table, index=classifiers,
                                        columns=table.columns.levels[2])
        # TODO check that performing the ranking of the rankings is appropriate
        str_table = to_latex(classifiers, df_mean_rankings, precision=1,
                             table_size=table_size,
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

        ## --------------------------------------------------------------##
        ## Version 2 for the aggregated rankings
        # Perform rankings of dataset+classifier vs calibration method
        table = df.pivot_table(index=['dataset', 'classifier'],
                               columns=['method'],
                               values=[measure], aggfunc=np.mean)
        # Remove datasets and classifier combinations in which one of the experiments failed
        table = table[~table.isna().any(axis=1)]
        if max_is_better:
            table *= -1
        ranking_table_all = table.apply(rankdata, axis=1).mean()
        ftest = compute_friedmanchisquare(table)
        print('Friedman test on the full table of shape {}'.format(
                    np.shape(table)))
        print(ftest)
        print(('Critical Difference V.2 computed with avranks of shape {} for' +
               '{} datasets').format(np.shape(ranking_table_all),
                                     len(table)))
        export_critical_difference(avranks=ranking_table_all,
                                   num_datasets=len(table),
                                   names=ranking_table_all.index.levels[1],
                                   filename=os.path.join(summary_path,
                                                         'crit_diff_' +
                                                         measure + '_v2.pdf'),
                                   title='(p-value = {:.2e}, #D = {})'.format(ftest.pvalue, len(table)))
        ## End Version 2 for the aggregated rankings
        ## --------------------------------------------------------------##


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
