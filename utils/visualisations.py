import matplotlib.pyplot as plt
import numpy as np

def df_to_heatmap(df, filename, title=None, figsize=None, annotate=True):
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

    yticklabels = [''.join(row).strip() for row in df.index.values]
    xticklabels = [''.join(col).strip() for col in df.columns.values]
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
    cax = ax.pcolor(df)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(xticklabels, rotation = 45, ha="right")

    middle_value = (df.values.max() + df.values.min())/2.0
    print(middle_value)
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
