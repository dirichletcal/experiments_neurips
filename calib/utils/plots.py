from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import beta

from calib.utils.functions import fit_beta_moments
from calib.utils.functions import df_normalise

import pandas as pd


def reliability_diagram(prob, Y, marker='--', label='', alpha=1, linewidth=1,
                        ax_reliability=None, clip=True):
    '''
        alpha= Laplace correction, default add-one smoothing
    '''
    bins = np.linspace(0,1+1e-16,11)
    prob = np.clip(prob, 0, 1)
    hist_tot = np.histogram(prob, bins=bins)
    hist_pos = np.histogram(prob[Y == 1], bins=bins)
    # Compute the centroids of every bin
    centroids = [np.mean(np.append(
                 prob[np.where(np.logical_and(prob >= bins[i],
                                              prob < bins[i+1]))],
                 bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]

    proportion = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+alpha*2)
    if ax_reliability is None:
        ax_reliability = plt.subplot(111)

    ax_reliability.plot(centroids, proportion, marker, linewidth=linewidth,
                        label=label)


def plot_reliability_diagram(scores_set, labels, legend_set,
                             original_first=False, alpha=1, **kwargs):
    fig_reliability = plt.figure('reliability_diagram')
    fig_reliability.clf()
    ax_reliability = plt.subplot(111)
    ax = ax_reliability
    # ax.set_title('Reliability diagram')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(scores_set[0], bins=bins)
        hist_pos = np.histogram(scores_set[0][labels == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            reliability_diagram(scores, labels, marker='x-',
                    label=legend, linewidth=n_lines, alpha=alpha, **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(scores_set[0], labels, 'kx', label=legend_set[0],
                markersize=9, markeredgewidth=1)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability


def plot_reliability_map(scores_set, prob, legend_set,
                         original_first=False, alpha=1, **kwargs):
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    # ax.set_title('Reliability map')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(prob[0], bins=bins)
        hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            if legend == 'uncalib':
                ax.plot([np.nan], [np.nan], '-', linewidth=n_lines,
                        **kwargs)
            else:
                ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                        **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(prob[0], prob[1], 'kx',
                label=legend_set[0], markersize=9, markeredgewidth=1)
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability_map


def plot_niculescu_mizil_map(scores_set, prob, legend_set, alpha=1, **kwargs):
    from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel((r'$s$'), fontsize=16)
    ax.set_ylabel((r'$\hat{p}$'), fontsize=16)
    #n_lines = len(legend_set)
    line_widths = {'beta': 3, 'isotonic': 1, 'logistic': 1}
    bins = np.linspace(0, 1, 11)
    hist_tot = np.histogram(prob[0], bins=bins)
    hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
    centers = (bins[:-1] + bins[1:])/2.0
    empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
    ax.plot(centers, empirical_p, 'ko', label='empirical')

    for (scores, legend) in zip(scores_set, legend_set):
        if legend != 'uncalib':
            ax.plot(prob[2], scores, '-', label=legend,
                    linewidth=line_widths[legend],
                    **kwargs)
        #n_lines -= 1
    ax.legend(loc='upper left')
    return fig_reliability_map


def remove_low_high(a1, a2, low, high):
    idx1 = np.logical_and(a1 > low, a1 <= high)
    idx2 = np.logical_and(a2 > low, a2 <= high)
    idx = np.logical_and(idx1, idx2)
    return a1[idx], a2[idx]


def plot_score_differences(dist, trained, parameter, limits=None):
    if limits is not None:
        dist, trained = remove_low_high(dist, trained, limits[0], limits[1])

    fig_score_differences = plt.figure('score_differences')
    fig_score_differences.clf()
    ax_score_differences = plt.subplot(111)
    ax = ax_score_differences
    ax.set_xlabel(parameter + '_dist', fontsize=16)
    ax.set_ylabel(parameter + '_trained', fontsize=16)
    ax.set_color_cycle(['black', 'red', 'blue'])
    ax.plot(dist, trained, 'o')
    iso = IsotonicRegression()
    iso.fit(dist, trained)
    scores = np.linspace(np.amin(dist), np.amax(dist), 10000)
    ax.plot(scores, iso.predict(scores))
    linear = LinearRegression()
    linear.fit(dist.reshape(-1, 1), trained.reshape(-1, 1))
    ax.plot(scores.reshape(-1, 1), linear.predict(scores.reshape(-1, 1)))
    return fig_score_differences


def plot_score_distributions(scores_pos, scores_neg, calibrator):
    fig_score_distributions = plt.figure('score_distributions')
    fig_score_distributions.clf()
    ax = plt.subplot(111)
    ax.set_color_cycle(['blue', 'red'])
    x_pos, bins_pos, p_pos = ax.hist(scores_pos, color='blue', alpha=0.3,
                                     label=r'hist$^+$', range=[0, 1],
                                     normed=True)
    x_neg, bins_neg, p_neg = ax.hist(scores_neg, color='red', alpha=0.3,
                                     label=r'hist$^-$', range=[0, 1],
                                     normed=True)
    factor = max(np.amax(x_pos), np.amax(x_neg))
    for item in p_pos:
        item.set_height(item.get_height() / factor)
    for item in p_neg:
        item.set_height(item.get_height() / factor)
    al_pos, bt_pos = fit_beta_moments(scores_pos)
    al_pos = np.clip(al_pos, 1e-16, np.inf)
    bt_pos = np.clip(bt_pos, 1e-16, np.inf)
    al_neg, bt_neg = fit_beta_moments(scores_neg)
    al_neg = np.clip(al_neg, 1e-16, np.inf)
    bt_neg = np.clip(bt_neg, 1e-16, np.inf)
    scores = np.linspace(0.0, 1.0, 1000)
    pdf_pos = beta.pdf(scores, al_pos, bt_pos)
    ax.plot(scores, pdf_pos/factor, linestyle=":",
            label='p(x|+)')
    pdf_neg = beta.pdf(scores, al_neg, bt_neg)
    ax.plot(scores, pdf_neg/factor, linestyle=":",
            label='p(x|-)')

    prior_pos = len(scores_pos) / (len(scores_pos) + len(scores_neg))
    prior_neg = len(scores_neg) / (len(scores_pos) + len(scores_neg))
    denominator = pdf_pos * prior_pos + pdf_neg * prior_neg
    prob_pos = (pdf_pos * prior_pos) / denominator
    prob_neg = (pdf_neg * prior_neg) / denominator
    ax.plot(scores, prob_pos, linestyle="--", label=r'p(+|x)$_{separate}$')
    ax.plot(scores, prob_neg, linestyle="--", label=r'p(-|x)$_{separate}$')
    pr_pos = calibrator.predict(scores)
    pr_neg = 1.0 - pr_pos
    ax.plot(scores, pr_pos, label='p(+|x)$_{betacal}$')
    ax.plot(scores, pr_neg, label='p(-|x)$_{betacal}$')
    ax.set_xlim([-0.001, 1.001])
    ax.set_ylim([0, 2.1])
    ax.legend(loc='upper right')
    return fig_score_distributions


# def sigmoid(x):
#     return np.exp(x) / (1 + np.exp(x))
#
#
# if __name__ == '__main__':
#     from sklearn.linear_model import LogisticRegression
#     np.random.seed(42)
#     # Random scores
#     n = np.random.normal(loc=-4, scale=2, size=100)
#     p = np.random.normal(loc=4, scale=2, size=100)
#     s = np.append(n, p)
#     plt.hist(s)
#     plt.show()
#     s.sort()
#     s1 = s.reshape(-1, 1)
#
#     # Obtaining probabilities from the scores
#     s1 = sigmoid(s1)
#     # Obtaining the two features for beta-calibration with 3 parameters
#     s1 = np.log(np.hstack((s1, 1.0 - s1)))
#     # s1[:, 1] *= -1
#
#     # Generating random labels
#     y = np.append(np.random.binomial(1, 0.1, 40), np.random.binomial(1, 0.3,
#                                                                      40))
#     y = np.append(y, np.random.binomial(1, 0.4, 40))
#     y = np.append(y, np.random.binomial(1, 0.4, 40))
#     y = np.append(y, np.ones(40))
#
#     # Fitting Logistic Regression without regularization
#     lr = LogisticRegression(C=99999999999)
#     lr.fit(s1, y)
#
#     linspace = np.linspace(-10, 10, 100)
#     l = sigmoid(linspace).reshape(-1, 1)
#     l1 = np.log(np.hstack((l, 1.0 - l)))
#     # l1[:, 1] *= -1
#
#     probas = lr.predict_proba(l1)[:, 1]
#     s_exp = sigmoid(s)
#     fig_map = plot_niculescu_mizil_map([probas], [s_exp, y, l],
#                                        ['beta'], alpha=1)
#
#     plt.show()

def multiindex_to_strings(index):
    if isinstance(index, pd.MultiIndex):
        return [' '.join(col).strip() for col in index.values]
    return [''.join(col).strip() for col in index.values]


def df_to_heatmap(df, filename, title=None, figsize=None, annotate=True,
                 normalise_columns=False, normalise_rows=False, cmap=None):
    ''' Exports a heatmap of the given pandas DataFrame

    Parameters
    ----------
    df:     pandas.DataFrame
        It should be a matrix, it can have multiple index and these will be
        flattened.

    filename: string
        Full path and filename to save the figure (including extension)

    title: string
        Title of the figure

    figsize:    tuple of ints (x, y)
        Figure size in inches

    annotate:   bool
        If true, adds numbers inside each box
    '''
    if normalise_columns:
        df = df_normalise(df, columns=True)
    if normalise_rows:
        df = df_normalise(df, columns=False)

    yticklabels = multiindex_to_strings(df.index)
    xticklabels = multiindex_to_strings(df.columns)
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        point_inch_ratio = 72.
        n_rows = df.shape[0]
        font_size_pt = plt.rcParams['font.size']
        xlabel_space_pt = max([len(xlabel) for xlabel in xticklabels])
        fig_height_in = ((xlabel_space_pt + n_rows) * (font_size_pt + 3)) / point_inch_ratio

        n_cols = df.shape[1]
        fig_width_in = df.shape[1]+4
        ylabel_space_pt = max([len(ylabel) for ylabel in yticklabels])
        fig_width_in = ((ylabel_space_pt + (n_cols * 3) + 5) * (font_size_pt + 3)) / point_inch_ratio
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    cax = ax.pcolor(df, cmap=cmap)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(xticklabels, rotation = 45, ha="right")

    middle_value = (df.max().max() + df.min().min())/2.0
    if annotate:
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                color = 'white' if middle_value > df.values[y, x] else 'black'
                plt.text(x + 0.5, y + 0.5, '%.2f' % df.values[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=color
                         )
    try:
        fig.tight_layout()
    except ValueError as e:
        print(e)
        print('Canceling tight_layout for figure {}'.format(filename))
    fig.savefig(filename)
    plt.close(fig)

def export_critical_difference(avranks, num_datasets, names, filename,
                               title=None, test='bonferroni-dunn'):
    '''
        test: string in ['nemenyi', 'bonferroni-dunn']
         - nemenyi two-tailed test (up to 20 methods)
         - bonferroni-dunn one-tailed test (only up to 10 methods)

    '''
    # Critical difference plot
    import Orange

    if len(avranks) > 10:
        print('Forcing Nemenyi Critical difference')
        test = 'nemenyi'
    cd = Orange.evaluation.compute_CD(avranks, num_datasets, alpha='0.05',
                                      test=test)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6,
                                  textspace=1.5)
    fig = plt.gcf()
    fig.suptitle(title, horizontalalignment='left')
    plt.savefig(filename)
    plt.close()


def export_boxplot(method, scores, y_test, n_classes, name_classes,
                   title, filename, file_ext ='.svg', per_class=True,
                   figsize=(4, 8), fig=None, ax=None):

    if fig is None and ax is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        fig.suptitle(title, bbox=dict(facecolor='white', lw=0))

    if not per_class:
        positive_scores = [scores[y_test == i,i].flatten() for i in
                           range(n_classes)]

        ax.set_ylim([0, 1])
        ebars = ax.errorbar(x=np.arange(1,n_classes+1)+0.4, y=[s.mean()
                                                                for s in
                                                                positive_scores],
                             yerr=[s.std() for s in positive_scores],
                             fmt='.', capsize=3, color='green',
                             clip_on=True, zorder=-900)
        ax.boxplot(positive_scores, notch=True, whis=[5, 95],
                   flierprops=dict(marker='o', markersize=3,
                                   markeredgewidth=.2))
        ax.set_xticklabels(name_classes, rotation=45, ha="right")

    else:
        fig = plt.figure(figsize=figsize)
        class_scores = [scores[y_test == i] for i in range(n_classes)]
        for i, cs_i in enumerate(class_scores):
            ax = fig.add_subplot(n_classes,1,i+1)
            ax.set_ylabel(name_classes[i])
            ax.set_ylim([0, 1])
            ebars = ax.errorbar(x=np.arange(1,n_classes+1)+0.4, y=cs_i.mean(axis=0),
                                 yerr=cs_i.std(axis=0),
                                 fmt='.', capsize=3, color='green',
                                 clip_on=True, zorder=-900)
            ax.boxplot(cs_i, notch=True, whis=[5, 95],
                       flierprops=dict(marker='o', markersize=3,
                                       markeredgewidth=.2))
            if i == n_classes-1:
                ax.set_xticklabels(name_classes, rotation=45, ha="right")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(filename + file_ext)
    plt.close(fig)


def export_dataset_analysis(df, measure, filename, file_ext='.svg'):
    df = df.copy()
    df['method'] = df['method'].astype('category')
    df['dataset'] = df['dataset'].astype('category')
    df['classifier'] = df['classifier'].astype('category')

    for method in df['method'].cat.categories:
        fig = plt.figure(figsize=(10,5))
        fig.suptitle(method)
        ax = fig.add_subplot(121)
        df[df['method'] == method].plot(kind='scatter', x='n_samples',
                                        y=measure, ax=ax, logx=True,
                                        alpha=0.5)
        ax = fig.add_subplot(122)
        df[df['method'] == method].plot(kind='scatter', x='n_classes',
                                        logx=True, y=measure, ax=ax, alpha=0.5)
        fig.savefig('{}_{}_{}.{}'.format(filename, method, measure, file_ext))
        plt.close(fig)


def plot_multiclass_reliability_diagram(y_true, p_pred, n_bins=15, title=None,
                                        fig=None, ax=None, legend=True,
                                        labels=['True class prop.',
                                                'Gap pred.  mean']):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(4, 2))
    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    y_true = y_true.flatten()
    p_pred = p_pred.flatten()

    bin_size = 1/n_bins
    centers = np.linspace(bin_size/2, 1.0 - bin_size/2, n_bins)
    true_proportion = np.zeros(n_bins)
    pred_mean = np.zeros(n_bins)
    for i, center in enumerate(centers):
        if i == 0:
            # First bin include lower bound
            bin_indices = np.where(np.logical_and(p_pred >= center - bin_size/2, p_pred <= center + bin_size/2))
        else:
            bin_indices = np.where(np.logical_and(p_pred > center - bin_size/2, p_pred <= center + bin_size/2))
        true_proportion[i] = np.mean(y_true[bin_indices])
        pred_mean[i] = np.mean(p_pred[bin_indices])

    ax.bar(centers, true_proportion, width=bin_size, edgecolor = "black",
           color = "blue", label=labels[0])
    ax.bar(centers, true_proportion - pred_mean,  bottom = pred_mean, width=bin_size/2,
           edgecolor = "red", color = "#ffc8c6", alpha = 1, label=labels[1])
    if legend:
        ax.legend()
    ax.plot([0,1], [0,1], linestyle = "--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig

def plot_reliability_diagram_per_class(y_true, p_pred, fig=None, ax=None, **kwargs):
    n_classes = y_true.shape[1]

    if fig is None and ax is None:
        fig = plt.figure(figsize=(n_classes*4, 2))

    if ax is None:
        ax = [fig.add_subplot(1, n_classes, i+1) for i in range(n_classes)]
    legend = True
    for i in range(n_classes):
        plot_multiclass_reliability_diagram(y_true[:,i], p_pred[:,i],
                                            title=r'$C_{}$'.format(i+1),
                                            fig=fig, ax=ax[i], legend=legend,
                                            **kwargs)
        legend = False
    return fig

def save_fig_close(fig, filename):
    fig.savefig(filename)
    plt.close(fig)
