import matplotlib.pyplot as plt
import numpy as np

def df_to_heatmap(df, filename, title=None, figsize=(6,4), annotate=True):
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
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    cax = ax.pcolor(df)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels([' '.join(row).strip() for row in df.index.values])
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels([' '.join(col).strip() for col in df.columns.values],
                       rotation = 45, ha="right")

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
    fig.tight_layout()
    fig.savefig(filename)
