from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


def trans_mat(trans_mat,
              states,
              include_values=True,
              xticks_rotation='horizontal',
              values_format=None,
              cmap='viridis',
              ax=None,
              colorbar=True):
    """plot transition matrix returned by
    ``article.bfbehav.sequence.transmat_from_annots``

    Parameters
    ----------
    trans_mat : numpy.array
    states : set
        of character labels
    include_values : bool
    xticks_rotation : str
        default is 'horizontal'
    values_format : str
        format string to use
        if including values.
        Default is None.
    cmap : str
        name of colormap,
        default is 'viridis'.
    ax : matplotlib.axes.Axes
        instance. Default is None,
        in which case a new
        figure is created
    colorbar : bool
        if True, add a colorbar

    Returns
    -------
    fig, ax
    """
    # adopted from
    # https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_plot/confusion_matrix.py#L168
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n_classes = len(states)
    im_ = ax.imshow(trans_mat, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(trans_mat, dtype=object)

        # print text with appropriate color depending on background
        thresh = (trans_mat.max() + trans_mat.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if trans_mat[i, j] < thresh else cmap_min

            if values_format is None:
                text_tm = format(trans_mat[i, j], '.2g')
                if trans_mat.dtype.kind != 'f':
                    text_d = format(trans_mat[i, j], 'd')
                    if len(text_d) < len(text_tm):
                        text_tm = text_d
            else:
                text_tm = format(trans_mat[i, j], values_format)

            text_[i, j] = ax.text(
                j, i, text_tm,
                ha="center", va="center",
                color=color)

    if colorbar:
        axins = inset_axes(ax,
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
        fig.colorbar(im_, cax=axins)

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=states,
           yticklabels=states,
           ylabel="From state",
           xlabel="To state")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    return fig, ax


def trans_mat_all_days(animal_day_transmats,
                       animal_id,
                       dpi=200,
                       figsize=(8, 4)):
    """plot transition matrices for multiple days

    Parameters
    ----------
    animal_day_transmats : dict
        dictionary of dictionaries
        where the value of
        ``animal_day_transmats[animal_id][datestr]``
        will be a TransitionMatrix named tuple,
        returned by ``article.bfbehav.sequence.transmat_from_annots``
    animal_id : str
        animal id that should be plotted, e.g. 'bl26lb16'
    dpi : int
        passed to ``plt.subplots`` when creating figure.
        Default is 200.
    figsize : tuple
        (width, height) in inches. Default is (8, 4).

    Returns
    -------
    None
    """
    day_transmats = animal_day_transmats[animal_id]
    ncol = len(day_transmats)

    fig, ax = plt.subplots(1, ncol, dpi=dpi, figsize=figsize)
    ax = ax.ravel()

    for ind, (day, trans_mat_tup) in enumerate(day_transmats.items()):
        if ind == len(day_transmats) - 1:
            colorbar = True
        else:
            colorbar = False
        mat, states = trans_mat_tup.matrix, trans_mat_tup.states
        # below ``trans_mat`` is function above we call to plot each
        trans_mat(mat, states, ax=ax[ind], colorbar=colorbar)
        ax[ind].set_title(day)
