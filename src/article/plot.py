"""functions to create plots, basis for figures in article"""
import matplotlib.pyplot as plt
import seaborn as sns


def frame_error(df, ax=None, figsize=(10, 3.75), legend=True, xlabel=False,
                save_as=None, transparent=True):
    """plot frame error results from generating a learning curve

    Parameters
    ----------
    df : pandas.DataFrame
        returned by article.util.make_df_birdsong_rec.
    figsize : tuple
        size of figure: (width, height) in inches.
        Default is (10, 3.75).
    xlabel : bool
        if True, add "training set size" label to x axis.
        Default is False (used when this function is called
        by error_rate_test_mean function).
    legend : bool
        if True, add legend to axis 0 (plot of individuals).
        Default is True.
    ax : matplotlib.axes.Axes
        axes on which to plot. Must be an array of size (1, 2).
        Default is None, in which case a new figure is created
        of size figsize, with the correct number of axes.
    save_as : str
        filename to save figure as
    transparent : bool
        if True, save figure as transparent. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        either same as axes passed by user, or new axes created
        (when axes is None).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        fig.set_size_inches(figsize)

    df['frame_error_test_mean_pct'] = df['frame_error_test_mean'] * 100
    sns.pointplot(x="train_set_dur", y="frame_error_test_mean_pct",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  scale=0.75,
                  ax=ax[0])
    if xlabel:
        ax[0].set_xlabel("training set duration (s)", fontsize=20)
    else:
        ax[0].set_xlabel("")
    ax[0].set_ylabel("frame error (%)", fontsize=20)
    ax[0].set_ylim([0., 20.])
    ax[0].tick_params(labelsize=16)
    ax[0].legend(bbox_to_anchor=(1.01, 1), loc=2,
                 borderaxespad=0.)

    sns.boxplot(x="train_set_dur", y="frame_error_test_mean_pct", data=df, ax=ax[1])
    if xlabel:
        ax[1].set_xlabel("training set duration (s)", fontsize=20)
    else:
        ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    ax[1].tick_params(labelsize=16)
    ax[1].set_ylim([0., 20.])

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', transparent=transparent)

    return ax


def syllable_error_rate(df, ax=None, figsize=(10, 3.75), legend=False,
                        save_as=None, transparent=True):
    """plot syllable error rate results from generating a learning curve

    Parameters
    ----------
    df : pandas.DataFrame
        returned by article.util.make_df_birdsong_rec.
    figsize : tuple
        size of figure: (width, height) in inches.
        Default is (10, 3.75).
    legend : bool
        if True, add legend to axis 0 (plot of individuals).
        Default is True.
    ax : matplotlib.axes.Axes
        axes on which to plot. Must be an array of size (1, 2).
        Default is None, in which case a new figure is created
        of size figsize, with the correct number of axes.
    save_as : str
        filename to save figure as
    transparent : bool
        if True, save figure as transparent. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        either same as axes passed by user, or new axes created
        (when axes is None).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        fig.set_size_inches(figsize)

    sns.pointplot(x="train_set_dur", y="syllable_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  ax=ax[0])
    ax[0].set_xlabel("training set duration (s)", fontsize=20)
    ax[0].set_ylabel("syllable error rate\n(distance)", fontsize=20)
    ax[0].set_ylim([0., 1.5])
    ax[0].tick_params(labelsize=16)
    if legend:
        ax[0].legend(bbox_to_anchor=(1.01, 1), loc=2,
                     borderaxespad=0.)
    else:
        ax[0].legend_.remove()

    sns.boxplot(x="train_set_dur", y="syllable_error_test_mean", data=df, ax=ax[1])
    ax[1].set_xlabel("training set duration (s)", fontsize=20)
    ax[1].set_ylabel("")
    ax[1].set_ylim([0., 1.5])
    ax[1].tick_params(labelsize=16)
    label = 'Koumura\nOkanoya 2016'
    handle = ax[1].scatter([1, 2], [0.84, 0.46], s=75, color='k', marker='s',
                           label=label)
    ax[1].legend([handle], [label])

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', transparent=transparent)

    return ax


def error_rate_test_mean(df, ax=None, figsize=(10, 7.5),
                         save_as=None, transparent=True):
    """plot both frame error and syllable error rate results from
    generating a learning curve.

    This function uses both functions defined above
    to reproduce the plots for the figure in the paper.

    Parameters
    ----------
    df : pandas.DataFrame
        returned by article.util.make_df_birdsong_rec.
    figsize : tuple
        size of figure: (width, height) in inches.
        Default is (10, 7.5).
    ax : matplotlib.axes.Axes
        axes on which to plot. Must be an array of size (2, 2).
        Default is None, in which case a new figure is created
        of size figsize, with the correct number of axes.
    save_as : str
        filename to save figure as
    transparent : bool
        if True, save figure as transparent. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        either same as axes passed by user, or new axes created
        (when axes is None).s
    """
    if ax is None:
        fig, ax = plt.subplots(2, 2, constrained_layout=True)
        fig.set_size_inches(figsize)

    frame_error(df, ax[0, :], legend=True)
    syllable_error_rate(df, ax[1, :], legend=False)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight', transparent=transparent)

    return ax
