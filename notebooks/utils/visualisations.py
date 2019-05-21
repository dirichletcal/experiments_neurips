import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import OneHotEncoder


def plot_reliability_diagram(score, labels, linspace, scores_set, legend_set,
                             laplace_reg=1, scatter_prop=0.0, fig=None, n_bins=10,
                             bins_count=True, title=None, diagonal=True,
                             **kwargs):
    '''
    Parameters
    ==========
    scores_set : list of array_like of floats
        List of scores given by different methods, the first one is always the
        original one
    labels : array_like of ints
        Labels corresponding to the scores
    legend_set : list of strings
        Description of each array in the scores_set
    laplace_reg : float
        Laplace regularization when computing the elements in the bins
    scatter_prop : float
        If original first specifies the proportion of points (score, label) to
        show
    fig : matplotlib.pyplot.figure
        Plots the axis in the given figure
    bins_count : bool
        If True, show the number of samples in each bin

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    '''
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)

    n_lines = len(legend_set)

    # Draw the empirical values in a histogram style
    # TODO careful that now the min and max depend on the scores
    s_min = min(score)
    s_max = max(score)
    bins = np.linspace(s_min, s_max, n_bins+1)
    hist_tot = np.histogram(score, bins=bins)
    hist_pos = np.histogram(score[labels == 1], bins=bins)
    edges = np.insert(bins, np.arange(len(bins)), bins)
    empirical_p = np.true_divide(hist_pos[0]+laplace_reg,
                                 hist_tot[0]+2*laplace_reg)
    empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                            empirical_p)
    p = plt.plot(edges[1:-1], empirical_p, label='original')
    # Draw the centroids of each bin
    centroids = [np.mean(np.append(
                 score[np.where(np.logical_and(score >= bins[i],
                                               score < bins[i+1]))],
                 bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]
    proportion = np.true_divide(hist_pos[0]+laplace_reg,
                                hist_tot[0]+laplace_reg*2)
    plt.plot(centroids, proportion, 'o', color=p[-1].get_color(), linewidth=2,
             label='centroid')
    for (x, y, text) in zip(centroids, proportion, hist_tot[0]):
        if y < 0.95:
            y += 0.05
        else:
            y -= 0.05
        plt.text(x, y, text, horizontalalignment='center',
                 verticalalignment='center')

    # Draw the rest of the lines
    for (scores, legend) in zip(scores_set, legend_set):
        # reliability_diagram(scores, labels, marker='o-', label=legend,
        #                     linewidth=n_lines, alpha=alpha, n_bins=n_bins,
        #                     **kwargs)
        plt.plot(linspace, scores, label=legend, linewidth=n_lines, **kwargs)
        n_lines -= 1

    # Draw some samples with the labels
    if scatter_prop:
        n_points = int(scatter_prop*len(labels))
        plt.plot(score[:n_points], labels[:n_points], 'kx',
                 label='samples ({:d}%)'.format(int(scatter_prop*100)),
                 markersize=6, markeredgewidth=1, alpha=0.4)

    if diagonal:
        ax.plot([0, 1], [0, 1], 'r--')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$\hat p$')
    ax.legend(loc='lower right')
    ax.grid(True)

    return fig

def plot_multiclass_reliability_diagram(y_true, p_pred, n_bins=15, title=None,
                                        fig=None, ax=None, legend=True):
    '''
        y_true needs to be (n_samples, n_classes)
            - where n_classes may be 1, for a two class problem
    '''
    if fig is None and ax is None:
        fig = plt.figure()
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
           color = "blue", label='True class prop.')
    ax.bar(centers, true_proportion - pred_mean,  bottom = pred_mean, width=bin_size/2,
           edgecolor = "red", color = "#ffc8c6", alpha = 1, label='Gap pred. mean')

    if legend:
        ax.legend()

    ax.plot([0,1], [0,1], linestyle = "--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig

def plot_reliability_diagram_per_class(y_true, p_pred, fig=None, ax=None, **kwargs):
    if fig is None and ax is None:
        fig = plt.figure()

    if len(y_true.shape) == 1:
        y_true = OneHotEncoder(categories='auto').transform(y_true)

    n_classes = y_true.shape[1]
    if ax is None:
        ax = [fig.add_subplot(1, n_classes, i+1) for i in range(n_classes)]
    for i in range(n_classes):
        plot_multiclass_reliability_diagram(y_true[:,i], p_pred[:,i],
                                            title=r'$C_{}$'.format(i+1),
                                            fig=fig, ax=ax[i], **kwargs)
    return fig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fig=None, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

def plot_weight_matrix(weights, bias, classes, title='Weight matrix',
                       cmap=plt.cm.Greens, fig=None, ax=None, **kwargs):
    """
    This function prints and plots the weight matrix.
    """
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    matrix = np.hstack((weights, bias.reshape(-1, 1)))

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, **kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    tick_marks = np.arange(len(classes))
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xticks(np.append(tick_marks, len(classes)))
    ax.set_xticklabels(np.append(classes, 'c'))

    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    ax.set_ylabel('Class')
    fig.tight_layout()


def plot_individual_pdfs(class_dist, x_grid=None, y_grid=None,
                         grid_levels = 200, fig=None, title=None,
                         cmaps=None, grid=True):
    if fig is None:
        fig = plt.figure()

    if x_grid is None:
        x_grid = np.linspace(-8, 8, grid_levels)
    else:
        grid_levels = len(x_grid)

    if y_grid is None:
        y_grid = np.linspace(-8, 8, grid_levels)

    xx, yy = np.meshgrid(x_grid, y_grid)

    if cmaps is None:
        cmaps = [None]*len(class_dist.priors)

    for i, (p, d) in enumerate(zip(class_dist.priors, class_dist.distributions)):
        z = d.pdf(np.vstack([xx.flatten(), yy.flatten()]).T)

        ax = fig.add_subplot(1, len(class_dist.distributions), i+1)
        if title is None:
            ax.set_title('$P(Y={})={:.2f}$\n{}'.format(i+1, p, str(d)), loc='left')
        else:
            ax.set_title(title[i])
        contour = ax.contourf(xx, yy, z.reshape(grid_levels,grid_levels),
                              cmap=cmaps[i])
        if grid:
            ax.grid()
        fig.colorbar(contour)
