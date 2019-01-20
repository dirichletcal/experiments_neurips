# Code is an adaptation from
# http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.tri as tri
from matplotlib import ticker

import pandas as pd


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    # Mid-points of triangle sides opposite of each corner
    midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0
                 for i in range(3)]

    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


def bc2xy(pvalues, corners):
    return np.dot(pvalues, corners)


def draw_tri_samples(pvals, classes, labels=None, fig=None, ax=None,
                     handles=None, grid=True, **kwargs):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    pvals = pvals[:,:3].copy()

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    if labels is None:
        labels = [r'$C_{}$'.format(i+1) for i in range(len(corners))]
    center = corners.mean(axis=0)
    for i, corner in enumerate(corners):
        text_x, text_y = corner - (center - corner)*0.1
        ax.text(text_x, text_y, labels[i], verticalalignment='center',
                horizontalalignment='center')

    xy = bc2xy(pvals, corners)
    ax.scatter(xy[:, 0], xy[:, 1], c=classes, **kwargs)

    if handles is not None:
        ax.legend(handles=handles)

    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.set_xbound(lower=-0.01, upper=1.01)
    ax.set_ybound(lower=-0.01, upper=(0.75**0.5)+0.01)
    ax.axis('off')

    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    plt.triplot(triangle, c='k', lw=0.5)

    if grid:
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=4)
        ax.triplot(trimesh, c='gray', lw=0.5)



def get_func_mesh_values(func, subdiv=8):
    '''
    Gets the values returned by the function func in a triangular mesh grid
    '''
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    vals = np.array([func(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)])
    return vals


def get_mesh_xy(subdiv=8):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    return zip(trimesh.x, trimesh.y)


def get_mesh_bc(**kwargs):
    mesh_xy = get_mesh_xy(**kwargs)
    mesh_bc = np.array([xy2bc(xy) for xy in mesh_xy])
    return mesh_bc


def draw_pdf_contours(dist, **kwargs):
    draw_func_contours(dist.pdf, **kwargs)


def draw_func_contours(func, labels=None, nlevels=200, subdiv=8, fig=None,
                       ax=None, grid=True, **kwargs):
    '''
    Parameters:
    -----------
    labels: None, string or list of strings
        If labels == 'auto' it shows the class number on each corner
        If labels is a list of strings it shows each string in the
        corresponding corner
        If None does not show any label
    '''
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    pvals = np.array([func(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)])

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    # FIXME I would like the following line to work, but the max value is not
    # shown. I had to do create manually the levels and increase the max value
    # by an epsilon. This could be a major problem if the epsilon is not small
    # for the original range of values
    # contour = ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
    # contour = ax.tricontourf(trimesh, pvals, nlevels, extend='both')
    contour = ax.tricontourf(trimesh, pvals,
                             levels=np.linspace(pvals.min(), pvals.max()+1e-9,
                                                nlevels),
                             **kwargs)

    # Colorbar
    cb = fig.colorbar(contour, ax=ax, fraction=0.1, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    # cb.ax.xaxis.set_major_locator(ticker.AutoLocator())
    cb.update_ticks()

    if labels is not None:
        if labels == 'auto':
            labels = [r'$C_{}$'.format(i+1) for i in range(len(corners))]
        center = corners.mean(axis=0)
        for i, corner in enumerate(corners):
            text_x, text_y = corner - (center - corner)*0.1
            ax.text(text_x, text_y, labels[i], verticalalignment='center',
                    horizontalalignment='center')

    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    ax.triplot(triangle, c='k', lw=0.8)

    if grid:
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=4)
        ax.triplot(trimesh, c='gray', lw=0.5)

    # Axes options
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=0.75**0.5)
    ax.set_xbound(lower=0, upper=1)
    ax.set_ybound(lower=0, upper=0.75**0.5)
    ax.axis('equal')
    ax.axis('off')
    plt.gca().set_adjustable("box")


def plot_individual_pdfs(class_dist, *args, **kwargs):
    fig = plt.figure(figsize=(16, 5))
    for i, (p, d) in enumerate(zip(class_dist.priors,
                                   class_dist.distributions)):
        ax = fig.add_subplot(1, len(class_dist.distributions), i+1)
        ax.set_title('$P(Y={})={}$\n$\\mathcal{{D}}_{}(\\alpha={})$'.format(
                         i+1, p, i+1, str(d)), loc='left')
        draw_pdf_contours(d, labels='auto', fig=fig, ax=ax, *args, **kwargs)


def plot_marginal(func, mesh, c, ax1, ax2):
    values = np.array([func(bc) for bc in mesh]).reshape(-1, 1)

    df = pd.DataFrame(np.concatenate((mesh, values), axis=1),
                      columns=['C1', 'C2', 'C3', 'P'])
    df.plot(kind='scatter', x=c, y='P', alpha=0.1, ax=ax1)

    ax2.set_title('Class {} marginal'.format(c))
    table = df.pivot_table(index=c, values='P')
    table.reset_index(inplace=True)
    table.columns = [c, 'P']
    table.plot(kind='scatter', x=c, y='P', alpha=0.2, ax=ax2)


def plot_converging_lines_pvalues(func, lines, i, ax):
    '''
    Plots the probability values of the given function for each given line.
    The i indicates the class index from 0 to 2
    '''
    # This orders the classes in the following manner:
    # C1, C2, C3
    # C2, C3, C1
    # C3, C1, C2
    classes = np.roll(np.array([0, 1, 2]), -i)

    for j, line in enumerate(lines):
        pvalues = np.array([func(p) for p in line]).flatten()
        ax.plot(line[:, i], pvalues,
                label=r'$C_{} = {}/{}, C_{} = {}/{}$'.format(
                    classes[1]+1, j, len(lines)-1,
                    classes[2]+1, len(lines)-j-1, len(lines)-1))
    ax.legend()


def get_converging_lines(num_lines, mesh_precision=10, class_index=0, tol=1e-6):
    '''
    If class_index = 0
    Create isometric lines from the oposite side of C1 simplex to the C1 corner
    First line has C2 fixed to 0
    Last line has C3 fixed to 0
          Class 3  line 1 start
                 /\
                /  \
               /    \ line 2 start
              /    - \
             /   -/   \
            /  -/      \
           / -/      ---\ line 3 start
          /-/  -----/    \
         //---/           \
        -------------------- line 4 start
    Class 1(lines end)      Class 2

    Else if class_index = [1, 2]
    Then the previusly described lines are rotated towards the indicated class.
    The lines always follow a clockwise order.
    '''
    p = np.linspace(0, 1, mesh_precision).reshape(-1, 1)
    q = np.linspace(0, 1, num_lines).reshape(-1, 1)
    lines = [np.hstack((p, (1-p)*q[i], (1-p)*(1-q[i]))) for i in range(len(q))]
    if class_index > 0:
        indices = np.array([0, 1, 2])
        lines = [line[:, np.roll(indices, class_index)] for i, line in
                 enumerate(lines)]
    return np.clip(lines, tol, 1.0 - tol)


def draw_calibration_map(original_p, calibrated_p, labels=None, fig=None, ax=None,
                     handles=None, subdiv=5, color=None, **kwargs):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    original_p = original_p[:,:3].copy()
    calibrated_p = calibrated_p[:,:3].copy()

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    if labels is None:
        labels = [r'$C_{}$'.format(i+1) for i in range(len(corners))]
    center = corners.mean(axis=0)
    for i, corner in enumerate(corners):
        text_x, text_y = corner - (center - corner)*0.1
        ax.text(text_x, text_y, labels[i], verticalalignment='center',
                horizontalalignment='center')

    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    ax.triplot(triangle, c='k', lw=0.8)

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    ax.triplot(trimesh, c='gray', lw=0.5)

    o_xy = bc2xy(original_p, corners)
    c_xy = bc2xy(calibrated_p, corners) - o_xy
    #ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
    ax.quiver(o_xy[:, 0], o_xy[:, 1], c_xy[:, 0], c_xy[:, 1], scale=1,
              color=color, angles='xy', **kwargs)

    if handles is not None:
        ax.legend(handles=handles)

    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.set_xbound(lower=-0.01, upper=1.01)
    ax.set_ybound(lower=-0.01, upper=(0.75**0.5)+0.01)
    ax.axis('off')

    return fig
