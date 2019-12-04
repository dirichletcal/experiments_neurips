import os
import re
import pandas as pd
import numpy as np
import math

from functools import partial

from calib.utils.functions import rankings_to_latex

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

    if df_list:
        df = pd.concat(df_list)
    else:
        df = pd.DataFrame()
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
    if table.shape[1] < 3:
        print('Friedman test not appropiate for less than 3 methods')
        class Ftest():
            def __init__(self, statistic, pvalue):
                self.statistic = statistic
                self.pvalue = pvalue
        return Ftest(np.nan, np.nan)

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
                              str_format='%.1f', position='tph'):
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


def summarise_confusion_matrices(df, summary_path, set_title=False,
        figsize=(16.5, 23.4)):
    '''
    figsize
        - (8.27, 11.69) for an A4
        - (11.69, 16.53) for an A3
        - (16.5, 23.4) for an A2
    '''

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
        fig = pyplot.figure(figsize=figsize) # (df_aux.shape[1]*3, df_aux.shape[0]*3))
        if set_title:
            fig.suptitle(calibrator)
        ij = 1
        for i, dat in enumerate(df_aux.index):
            for j, cla in enumerate(df_aux.columns.levels[1]):
                values = df_aux.loc[dat, ('confusion_matrix', cla)]
                ax = fig.add_subplot(len(df_aux), len(df_aux.columns), ij)
                if j == 0:
                    ax.set_ylabel(dat[:10])
                if i == 0:
                    ax.set_title(cla)
                ij += 1
                if values is None:
                    print('There are no confusion matrices for {}, {}, {}'.format(
                          calibrator, dat, cla))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
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
        fig.subplots_adjust(hspace = 0.0)
        fig.tight_layout()
        fig.savefig(os.path.join(summary_path,
            'confusion_matrices_{}.pdf'.format(calibrator)))


def summarise_hyperparameters(df, summary_path, set_title=False,
                              figsize=(16.5, 23.4)):
    '''
    figsize
        - (8.27, 11.69) for an A4
        - (11.69, 16.53) for an A3
        - (16.5, 23.4) for an A2
    '''
    def MakeList(x):
        T = tuple(x)
        if len(T) > 1:
            return T
        else:
            return T[0]

    # Histograms of parameters
    MAP_METHOD = {'OvR_Freq_Bin': 'n_bins=(?P<bins>\w+), ',
                  'OvR_Width_Bin': 'n_bins=(?P<bins>\w+), ',
                  'Dirichlet_L2': " 'l2': ([0-9\.\-e]+),",
                  'OvR_Beta_L2': " 'l2': ([0-9\.\-e]+),",
                  'dir_full_comp_l2': " 'l2': ([0-9\.\-e]+),",
                  'dirichlet_full_prefixdiag_l2': " 'l2': ([0-9\.\-e]+),",
                  'Log_Reg_L2': " 'C': ([0-9\.\-e]+),",
                  'mlr_logit': " 'C': ([0-9\.\-e]+),",
                  'OvR_Log_Reg_L2': " 'C': ([0-9\.\-e]+),",
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
        all_flat = df_aux.values.flatten()
        all_flat = all_flat[all_flat != None]
        all_flat = np.hstack(sum([aux for aux in all_flat if not
                                  isinstance(aux, float)], ()))
        all_unique = np.unique(all_flat)
        sorted_idx = np.argsort(all_unique)
        if (all_unique == np.floor(all_unique)).all():
            xticklabels = [str(int(x)) for x in all_unique[sorted_idx]]
        else:
            xticklabels = [np.format_float_scientific(x, precision=2) for x in
                           all_unique[sorted_idx]]
        print('Unique hyperparameters')
        print(all_unique)

        # Generate one barplot with all hyperparameters
        fig = pyplot.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
        uniq, counts = np.unique(all_flat, return_counts=True)
        sorted_idx = np.argsort(uniq)
        uniq = uniq[sorted_idx]
        counts = counts[sorted_idx]
        ax.bar(sorted_idx, counts)
        ax.set_xticks(sorted_idx)
        ax.set_xticklabels(xticklabels, rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(os.path.join(summary_path, 'bars_hyperparameters_all_{}.pdf'.format(key)))

        # Generate one barplot per dataset and classifier combination
        #fig = pyplot.figure(figsize=(df_aux.shape[1]*3, df_aux.shape[0]*3))
        fig = pyplot.figure(figsize=figsize)
        if set_title:
            fig.suptitle(key)
        ij = 0
        for i, dat in enumerate(df_aux.index):
            for j, cla in enumerate(df_aux.columns.levels[1]):
                ij += 1

                values = df_aux.loc[dat, ('calibrators', cla)]

                ax = fig.add_subplot(len(df_aux), len(df_aux.columns), ij)

                if isinstance(values, float) and math.isnan(values):
                    print('There are no hyperparameters for {}, {}, {}'.format(
                          key, dat, cla))
                    values = [[]]

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
                counts = counts[sorted_idx]
                ax.bar(sorted_idx, counts)
                ax.set_xticks(sorted_idx)
                if j == 0:
                    ax.set_ylabel(dat[:10])
                if i == 0:
                    ax.set_title(cla)
                if i == len(df_aux.index)-1:
                    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
                else:
                    ax.set_xticklabels([])

        fig.subplots_adjust(hspace = 0.0)
        fig.tight_layout()
        fig.savefig(os.path.join(summary_path, 'bars_hyperparameters_{}.pdf'.format(key)))

    # heatmaps of parameters
    # FIXME change MAP method as it is not being used
    def weight_matrix(string, restore_last_class=False):
        solution = re.findall("'weights_': array(.*?)]]\)", string, flags=re.DOTALL)
        matrices = []
        for s in solution:
            x = np.fromstring(''.join(c for c in s if c in
                                                  '0123456789.-e+,'), sep=',')
            x = x.reshape(int(np.floor(np.sqrt(len(x)))), -1)
            if restore_last_class:
                col_sums = np.sum(x,axis=0)
                amount_to_shift = ( col_sums[:-1] - np.diag(x) ) / (x.shape[0]-1)
                x = x - np.concatenate((amount_to_shift,[0]))
                x[:,-1] = x[:,-1] - col_sums[-1] / x.shape[0]
            matrices.append(x)
        return matrices

    weight_matrix_rlc = partial(weight_matrix, restore_last_class=True)

    def weight_matrix_theorem5(string):
        solution = re.findall("'weights_': array(.*?)]]\)", string, flags=re.DOTALL)
        matrices = []
        for s in solution:
            W = np.fromstring(''.join(c for c in s if c in
                                                  '0123456789.-e+,'), sep=',')
            W = W.reshape(int(np.floor(np.sqrt(len(W)))), -1)
            b = W[:, -1]
            W = W[:,:-1]
            col_min = np.min(W,axis=0)
            A = W - col_min
            softmax = lambda z:np.divide(np.exp(z), np.sum(np.exp(z)))
            c = softmax(np.matmul(W, np.log(np.ones(len(b))/len(b))) + b)

            matrices.append(np.hstack((A, c.reshape(-1,1))))
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

    MAP_METHOD = {'Dirichlet_L2': weight_matrix_theorem5,
                  'dir_keras': weights_keras,
                  'dir_full_gen': weight_matrix,
                  'dir_full_comp_l2': weight_matrix_theorem5,
                  'OvR_Beta': weight_matrix,
                  'dirichlet_full_prefixdiag_l2': weight_matrix_theorem5,
                  'mlr_log': coef_intercept_matrix,
                  'mlr_logit': coef_intercept_matrix,
                  'ovr_mlr_log': coef_intercept_matrix,
                  'ovr_mlr_logit': coef_intercept_matrix
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
                if isinstance(values, float) and math.isnan(values):
                    continue
                # Stacking (#iter x #crossval x #crossval) on first dimension
                parameters = np.concatenate(values).mean(axis=0)
                # Dirichlet Theorem5
                if key in ['Dirichlet_L2', 'dir_full_comp_l2',
                           'dirichlet_full_prefixdiag_l2']:
                    col_min = np.min(parameters,axis=0)[:-1]
                    parameters[:,:-1] = parameters[:,:-1] - col_min

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
        hyperparameters=True, confusion_matrices=True, 
                      reduced_names=True):
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
        - 'train_guo-ece': Training binary ECE score
        - 'train_cla-ece': Training classwise ECE score
        - 'train_full-ece': Training full ECE score
        - 'train_mce': Training MCE score
        - 'acc': Accuracy
        - 'loss': log-loss
        - 'brier': Brier score
        - 'guo-ece': Binary ECE score
        - 'cla-ece': Classwise ECE score
        - 'full-ece': Full ECE score
        - 'p-guo-ece': p-value Guo ECE score
        - 'p-cla-ece': p-value Classwise ECE score
        - 'p-full-ece': p-value Full ECE score
        - 'mce': MCE score
        - 'exec_time': Mean execution time
        - 'classifier': Original classifier used to train
        - 'calibrators': List of calibrators with their parameters

    '''
    # Change name of metrics
    df.rename({'guo-ece': 'conf-ece', 'p-guo-ece': 'p-conf-ece',
               'train_guo-ece': 'train_conf-ece',
               'cla-ece': 'cw-ece', 'p-cla-ece': 'p-cw-ece',
               'train_cla-ece': 'train_cw-ece'},
               axis='columns', inplace=True)

    # Shorten some names
    shorten = dict(dirichlet='dir', binning='bin', logistic='mlr',
                   uncalibrated='uncal')
    for key, value in shorten.items():
        df['method'] = df['method'].replace(to_replace=key, value=value,
                                        regex=True)
    # Names for final version
    if reduced_names:
        final_names = dict(
            dir_fix_diag='TempS',
            temperature_scaling='TempS',
            vector_scaling='VecS',
            uncal='Uncal',
            ovr_dir_full='Beta',
            bin_freq='FreqB',
            bin_width='WidthB',
            dir_full_l2='DirL2',
            isotonic='Isot',
            dir_full='Dir',
            mlr_log='MultLogRegL2',
            ovr_dir_full_l2='BetaL2',
            ovr_mlr_log='LogRegL2',
            dir_odir_l2='DirODIR',
            temperature_scaling_noref='TempSNoref',
            vector_scaling_noref='VecSNoref',
            dir_odir_l2_noref='DirODIRNoref',
            dir_full_noref='DirNoref',
            dir_full_l2_noref='DirL2Noref',
        )

        new_order = ['Uncal', 'DirL2', 'DirL2Noref', 'DirODIR', 'DirODIRNoref',
                     'Beta', 'TempS', 'TempSNoref', 'VecS', 'VecSNoref',
                     'Isot', 'FreqB', 'WidthB']
    else:
        final_names = dict(
            dir_fix_diag='Temp_Scaling',
            temperature_scaling='Temp_Scaling',
            vector_scaling='Vect_Scaling',
            uncal='Uncalibrated',
            ovr_dir_full='OvR_Beta',
            bin_freq='OvR_Freq_Bin',
            bin_width='OvR_Width_Bin',
            dir_full_l2='Dirichlet_L2',
            isotonic='OvR_Isotonic',
            dir_full='Dirichlet',
            mlr_log='Log_Reg_L2',
            ovr_dir_full_l2='OvR_Beta_L2',
            ovr_mlr_log='OvR_Log_Reg_L2',
            dir_odir_l2='Dirichlet_ODIR')

    for key, value in final_names.items():
        df['method'] = df['method'].replace(to_replace=key, value=value,
                                            regex=False)

    new_order = [method for method in new_order if method in
                 df['method'].unique()]

    dataset_names = df['dataset'].unique()
    classifiers = df['classifier'].unique()

    measures_list = ['acc', 'loss', 'brier', 'conf-ece', 'cw-ece', 'full-ece',
                     'p-conf-ece', 'p-cw-ece', 'p-full-ece', 'mce']
    measures_list = [measure for measure in measures_list if measure in df.columns]

    # Assert that all experiments have finished
    for column in ['method', 'classifier']:
        for measure in measures_list:
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

    measures_list = (('acc', True), ('loss', False), ('brier', False),
                     ('conf-ece', False), ('cw-ece', False),
                     ('full-ece', False), ('p-conf-ece', True),
                     ('p-cw-ece', True), ('p-full-ece', True),
                     ('mce', False), ('exec_time', False),
                     #('train_acc', True), ('train_loss', False),
                     #('train_brier', False), ('train_conf-ece', False),
                     #('train_cw-ece', False), ('train_full-ece', False),
                     #('train_mce', False)
                    )
    measures_list = [(key, value) for key, value in measures_list if key in
        df.columns]
    for measure, max_is_better in measures_list:
        print('# Measure = {}'.format(measure))
        if 'train_' not in measure:
            filename = os.path.join(summary_path,
                                'n_samples_scatter')
            export_dataset_analysis(df, measure, filename)

        table = df.pivot_table(index=['classifier'], columns=['method'],
                               values=[measure], aggfunc=[np.mean, np.std])

        table = table.reindex(new_order, axis=1, level='method')
        table.sort_index(inplace=True)

        str_table = rankings_to_latex(classifiers, table, precision=2,
                             table_size=table_size, max_is_better=max_is_better,
                             caption=('Ranking of calibration methods ' +
                                      'applied on different classifiers ' +
                                      'with the mean measure={}'
                                      ).format(measure),
                             label='table:mean:{}'.format(measure))

        table = df.pivot_table(index=['mc', 'test_fold', 'dataset',
                                      'classifier'], columns=['method'],
                               values=[measure], aggfunc=[len])
        table = table.reindex(new_order, axis=1, level='method')
        table.sort_index(inplace=True)

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

            table = table.reindex(new_order, axis=1, level='method')
            table.sort_index(inplace=True)

            table.columns = table.columns.droplevel()
            table.to_csv(os.path.join(summary_path, measure + '.csv'))

            # Print correlation results
            method = 'spearman' # for non-linear ranking correlations
            #method = 'pearson' # for linear ranking correlations
            print('\n{} correlation for the measure {}'.format(method, measure))
            corr_test = table.reset_index(level=['n_classes', 'n_samples']).corr(method=method)
            print(corr_test)

            if ('train_' + measure) in [m[0] for m in measures_list]:
                table = df.pivot_table(index=['dataset', 'classifier', 'n_classes',
                                              'n_samples'],
                                       columns=['method'],
                                       values=['train_' + measure])
                table = table.reindex(new_order, axis=1, level='method')
                table.sort_index(inplace=True)
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

        cmap = pyplot.get_cmap('tab20')
        if measure.startswith('p-'):
            _p_table_nonan = table.dropna(axis=0)
            _p_table = (_p_table_nonan > 0.05).mean(axis=0)
            _p_table.sort_values(ascending=max_is_better, inplace=True)
            _p_table.reset_index(level=0, drop=True, inplace=True)
            filename = os.path.join(summary_path,
                                    'p_table_calibrators_{}'.format(measure))
            _p_table.to_latex(filename + '.tex')
            fig = pyplot.figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            _p_table.plot(kind='barh', ax=ax, title=None, # '{} > 0.05'.format(measure),
                         zorder=2, color=cmap(_p_table.index.argsort().argsort()))
            ax.grid(zorder=0)
            ax.set_xlabel('Proportion (out of {})'.format(_p_table_nonan.shape[0]))
            pyplot.tight_layout()
            pyplot.savefig(filename + '.svg')
            pyplot.close(fig)

        print('Percentage of finite results per calibrator')
        print(np.isfinite(table.values).mean(axis=0))

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
        measure_table = np.zeros((len(classifiers),
                                  df.method.unique().shape[0]))
        num_datasets = np.zeros(len(classifiers), dtype='int')
        for i, classifier_name in enumerate(classifiers):
            print('- Classifier name = {}'.format(classifier_name))
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure],
                                               aggfunc=[np.mean, np.std])
            table = table.reindex(new_order, axis=1, level='method')
            table.sort_index(inplace=True)

            # Perform a Friedman statistic test
            # Remove datasets in which one of the experiments failed
            ftest = compute_friedmanchisquare(table['mean'])
            print(ftest)

            str_table = rankings_to_latex(dataset_names, table, precision=2,
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
            table = table.reindex(new_order, axis=1, level='method')
            table.sort_index(inplace=True)
            table = table[~table.isna().any(axis=1)]
            if max_is_better:
                table *= -1
            measure_table[i] = table['mean'].mean()
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
        ## 1.1. Export the summary of all rankings
        df_mean_rankings = pd.DataFrame(ranking_table, index=classifiers,
                                        columns=table.columns.levels[2])
        df_mean_measures = pd.DataFrame(measure_table, index=classifiers,
                                        columns=table.columns.levels[2])
        if max_is_better:
            df_mean_measures *= -1

        ## --------------------------------------------------------------##
        ## Version 2 for the aggregated rankings
        # Perform rankings of dataset+classifier vs calibration method
        table = df.pivot_table(index=['dataset', 'classifier'],
                               columns=['method'],
                               values=[measure], aggfunc=np.mean)
        table = table.reindex(new_order, axis=1, level='method')
        table.sort_index(inplace=True)
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
                                   names=table.columns.levels[1],
                                   filename=os.path.join(summary_path,
                                                         'crit_diff_' +
                                                         measure + '_v2.pdf'),
                                   title='(p-value = {:.2e}, #D = {})'.format(ftest.pvalue, len(table)))
        ## End Version 2 for the aggregated rankings
        ## --------------------------------------------------------------##

        ## 1.2. Export the summary of all rankings
        # TODO check that performing the ranking of the rankings is appropriate
        print('Average rankings shape = {}'.format(ranking_table_all.shape))
        print('Average rankings = {}'.format(ranking_table_all))
        #df_mean_rankings.rename(reduced_names, axis='columns', inplace=True)
        #df_mean_rankings = df_mean_rankings[new_order]
        df_mean_rankings.sort_index(inplace=True)
        str_table = rankings_to_latex(df_mean_rankings.index, df_mean_rankings,
                                      precision=1, table_size=table_size,
                             max_is_better=False,
                             caption=('Ranking of calibration methods ' +
                                      'for {} (Friedman\'s test significant ' +
                                      'with p-value {:.2e}'
                                      ).format(measure, ftest.pvalue),
                             label='table:{}'.format(measure),
                             add_std=False,
                             column_names=df_mean_rankings.columns,
                             avg_ranks=ranking_table_all, add_rank=False)
        file_basename = os.path.join(summary_path,
                                     '{}_rankings'.format(measure))
        with open(file_basename + '.tex', "w") as text_file:
            text_file.write(str_table)

        # First version of table with the average measures
        measure_table_all = df_mean_measures.mean(axis=0)
        print('Average measures = {}'.format(df_mean_measures))
        str_table = rankings_to_latex(classifiers, df_mean_measures,
                                      precision=2,
                             table_size=table_size,
                             max_is_better=max_is_better,
                             caption=('Ranking of calibration methods ' +
                                      'applied to each classifier ' +
                                      'with the measure={}'
                                      ).format(measure),
                             label='table:{}'.format(measure),
                             add_std=False,
                             column_names=measure_table_all.index,
                             avg_ranks=measure_table_all, add_rank=True)
        file_basename = os.path.join(summary_path,
                                     '{}_average'.format(measure))
        with open(file_basename + '.tex', "w") as text_file:
            text_file.write(str_table)

        # Create effect size measures
        ave_relative = np.zeros((len(classifiers),
                                df.method.unique().shape[0]))
        for i, classifier_name in enumerate(classifiers):
            print('- Classifier name = {}'.format(classifier_name))
            class_mask = df['classifier'] == classifier_name
            table = df[class_mask].pivot_table(index=['dataset'],
                                               columns=['method'],
                                               values=[measure])
            table = table.reindex(new_order, axis=1, level='method')
            table.sort_index(inplace=True)
            uncal_measure = table[(measure, 'Uncal')]
            table_values = (table.values -
                            uncal_measure.values.reshape(-1,1)
                           )/uncal_measure.values.reshape(-1,1)
            table.iloc[:,:] = table_values

            str_table = rankings_to_latex(table.index, table, precision=3,
                                 table_size=table_size,
                                 max_is_better=max_is_better,
                                 caption=('Ranking of calibration methods ' +
                                          'applied on the classifier ' +
                                          '{} with the relative measure={}'
                                          ).format(classifier_name, measure),
                                 label='table:rel:{}:{}'.format(classifier_name,
                                                            measure),
                                 column_names=table.columns.levels[1],
                                 add_std=False)
            file_basename = os.path.join(summary_path, classifier_name +
                                         '_dataset_vs_method_relative_' + measure)
            with open(file_basename + '.tex', "w") as text_file:
                text_file.write(str_table)
            ave_relative[i] = table.mean(axis=0)
        df_ave_relative = pd.DataFrame(ave_relative, index=classifiers,
                                       columns=table.columns.levels[1])
        # First version of table with the average measures
        ave_relative_all = df_ave_relative.mean(axis=0)
        print('Average measures = {}'.format(ave_relative_all))
        str_table = rankings_to_latex(classifiers, df_ave_relative,
                                      precision=3,
                             table_size=table_size,
                             max_is_better=max_is_better,
                             caption=('Ranking of calibration methods ' +
                                      'applied to each classifier ' +
                                      'with the relative measure={}'
                                      ).format(measure),
                             label='table:rel:{}'.format(measure),
                             add_std=False,
                             column_names=df_ave_relative.columns,
                             avg_ranks=ave_relative_all, add_rank=True)
        file_basename = os.path.join(summary_path,
                                     '{}_rel_average'.format(measure))
        with open(file_basename + '.tex', "w") as text_file:
            text_file.write(str_table)

        # Answering rebuttal
        base_name = 'Uncal'
        for measure, max_is_better in measures_list:
            table = df.pivot_table(index=['dataset', 'classifier', 'mc'],
                                   columns=['method'], values=[measure],
                                   )
            table = table.reindex(new_order, axis=1, level='method')
            table.sort_index(inplace=True)
            if measure == 'acc':
                table = 1 - table
            base_measure = table[(measure, base_name)]
            table_values = 100*(table.values - 
                                base_measure.values.reshape(-1,1)
                           )/base_measure.values.reshape(-1,1)
            table.iloc[:,:] = table_values

            #relative_improvement = table[(measure, 'Dirichlet_L2')].agg(['min', 'max', 'mean', 'median'])
            #print(measure)

            relative_improvement = table.agg(['min', 'max', 'mean', 'median'])
            print(relative_improvement)
            relative_improvement.to_latex(os.path.join(summary_path,
                                     '{}_relative_statistics.tex'.format(measure)))





    for classifier_name in classifiers:
        table = df.pivot_table(values=[key for key, value in measures_list],
                               index=['dataset', 'method'],
                               aggfunc=[np.mean, np.std])
        table.sort_index(inplace=True)
        table.to_csv(os.path.join(summary_path, classifier_name + '_main_results.csv'))
        table.to_latex(os.path.join(summary_path, classifier_name + '_main_results.tex'))


def generate_classifier_summaries(df, summary_path, table_size='small'):
    # Change name of metrics
    df.rename({'guo-ece': 'conf-ece', 'p-guo-ece': 'p-conf-ece',
               'train_guo-ece': 'train_conf-ece',
               'cla-ece': 'cw-ece', 'p-cla-ece': 'p-cw-ece',
               'train_cla-ece': 'train_cw-ece'},
               axis='columns', inplace=True)

    dataset_names = df['dataset'].unique()
    classifiers = df['classifier'].unique()

    df = df[df.method == 'uncalibrated']

    measures_list = ['acc', 'loss', 'brier', 'conf-ece', 'cw-ece', 'full-ece',
                     'p-conf-ece', 'p-cw-ece', 'p-full-ece', 'mce']
    measures_list = [measure for measure in measures_list if measure in df.columns]

    measures_list = (('acc', True), ('loss', False), ('brier', False),
                     ('conf-ece', False), ('cw-ece', False),
                     ('full-ece', False), ('p-conf-ece', True),
                     ('p-cw-ece', True), ('p-full-ece', True),
                     ('mce', False), ('train_acc', True),
                     ('train_loss', False), ('train_brier', False),
                     ('exec_time', False), ('train_conf-ece', False),
                     ('train_cw-ece', False), ('train_full-ece', False),
                     ('train_mce', False), ('exec_time', False))
    measures_list = [(key, value) for key, value in measures_list if key in
        df.columns]
    for measure, max_is_better in measures_list:
        print('# Measure = {}'.format(measure))

        table = df.pivot_table(index=['mc', 'test_fold', 'dataset',
                                      ], columns=['classifier'],
                               values=[measure])
        cmap = pyplot.get_cmap('tab20')
        if measure.startswith('p-'):
            _p_table_nonan = table.dropna(axis=0)
            _p_table = (_p_table_nonan > 0.05).mean(axis=0)
            _p_table.sort_values(ascending=max_is_better, inplace=True)
            _p_table.reset_index(level=0, drop=True, inplace=True)
            filename = os.path.join(summary_path,
                                    'p_table_classifiers_{}'.format(measure))
            _p_table.to_latex(filename + '.tex')
            fig = pyplot.figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            _p_table.plot(kind='barh', ax=ax, title=None, #'{} > 0.05'.format(measure),
                          zorder=2, color=cmap(_p_table.index.argsort().argsort()))
            ax.grid(zorder=0)
            ax.set_xlabel('Proportion (out of {})'.format(_p_table_nonan.shape[0]))
            pyplot.tight_layout()
            pyplot.savefig(filename + '.svg')
            pyplot.close(fig)


        print('Percentage of finite results per classifier')
        print(np.isfinite(table.values).mean(axis=0))

        # Wilcoxon rank-sum test two-tailed
        df_ranksums = compute_ranksums(table)

        filename = os.path.join(summary_path,
                                'classifiers_ranksum_pvalues_{}.tex'.format(measure))
        threshold = 0.005
        export_statistic_to_latex(df_ranksums, filename, threshold=threshold,
                                 caption=('Wilcoxon rank-sum test statistic '
                                          'for every paired uncalibrated '
                                          'classifier for the '
                                          'measure of {}. Statistic is bold '
                                          'when p-value is smaller than '
                                          '{}').format(measure, threshold),
                                 label='tab:ranksum:{}'.format(measure)
                                )

        # Mann-Whitney rank test one-sided alternative is first is smaller than
        df_mannwhitneyu = compute_mannwhitneyu(table)

        filename = os.path.join(summary_path,
                                'classifiers_mannwhitneyu_pvalues_{}.tex'.format(measure))
        export_statistic_to_latex(df_mannwhitneyu, filename, threshold=threshold,
                                  caption=('Mann-Whitney U test statistic '
                                           'one sided with alternative '
                                           'hypothesis the classifier in row i '
                                           'is less than the classifier in column j '
                                           'for every pair of uncalibrated '
                                           'classifiers for the '
                                           'measure of {}. Statistic is bold '
                                           'when the p-value is smaller than '
                                           '{}').format(measure, threshold),
                                  label='tab:mannwhitney:{}'.format(measure),
                                  str_format='%1.1e'
                                )

        table = df.pivot_table(index=['dataset'], columns=['classifier'],
                               values=[measure], aggfunc=[np.mean, np.std])
        table = table[~table.isna().any(axis=1)]
        ftest = compute_friedmanchisquare(table['mean'])
        print(ftest)
        str_table = rankings_to_latex(dataset_names, table, precision=2,
                             table_size=table_size,
                             max_is_better=max_is_better,
                             caption=('Ranking of uncalibrated classifiers ' +
                                      'with the measure={}' +
                                      '(Friedman statistic test ' +
                                      '= {:.2E}, p-value = {:.2E})'
                                      ).format(measure,
                                              ftest.statistic, ftest.pvalue),
                             label='table:{}:{}'.format('uncal', measure),
                             add_std=False)
        file_basename = os.path.join(summary_path,
                                     'dataset_vs_classifier_' + measure)
        with open(file_basename + '.tex', "w") as text_file:
            text_file.write(str_table)

        if max_is_better:
            table *= -1
        ranking_table = table['mean'].apply(rankdata, axis=1).mean()

        filename = os.path.join(summary_path, 'crit_diff_' +
                                'uncal_classifiers' + '_' +
                                measure + '.pdf')

        print(('Critical Difference computed with avranks of shape {} ' +
               'for {} datasets').format(np.shape(ranking_table),
                                     table.shape[0]))
        export_critical_difference(avranks=ranking_table,
                                   num_datasets=table.shape[0],
                                   names=table.columns.levels[2],
                                   filename=filename,
                                   title='(p-value = {:.2e}, #D = {})'.format(
                                       ftest.pvalue, table.shape[0]))


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
