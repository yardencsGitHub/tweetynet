"""functions to create plots, basis for figures in article"""
import matplotlib.pyplot as plt
import seaborn as sns


def frame_error_rate_test_mean(df, ax=None, figsize=(10, 7.5), save_as=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    sns.boxplot(x="train_set_dur", y="frame_error_test_mean", data=df, ax=ax)
    for i, box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        box.set_linewidth(4)

        # iterate over whiskers and median lines
        for j in range(6 * i, 6 * (i + 1)):
            ax.lines[j].set_color('black')
            ax.lines[j].set_linewidth(4)

    sns.pointplot(x="train_set_dur", y="frame_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  ax=ax)

    ax.set_ylabel("frame error", fontsize=28)
    ax.set_xlabel("training set duration (s)", fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2,
              borderaxespad=0., fontsize=20)
    ax.set_title("mean frame error per bird on test set\nv. training set duration", fontsize=30)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')

    return ax


def syllable_error_rate_test_mean(df, ax=None, figsize=(10, 7.5), save_as=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    sns.boxplot(x="train_set_dur", y="syllable_error_test_mean", data=df, ax=ax)
    for i, box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        box.set_linewidth(4)

        # iterate over whiskers and median lines
        for j in range(6 * i, 6 * (i + 1)):
            ax.lines[j].set_color('black')
            ax.lines[j].set_linewidth(4)

    sns.pointplot(x="train_set_dur", y="syllable_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  ax=ax)

    ax.set_ylabel("syllable error rate", fontsize=28)
    ax.set_xlabel("training set duration (s)", fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2,
              borderaxespad=0., fontsize=20)
    ax.set_title("mean syllable error rate per bird on test set\nv. training set duration", fontsize=30)

    ax.scatter(1, 0.84, s=75, color='k', marker='s')
    ax.text(0.75, 0.76, 'Koumura\nOkanoya 2016', fontsize=18)
    ax.scatter(2, 0.5, s=75, color='k', marker='s')
    ax.text(1.75, 0.52, 'Koumura\nOkanoya 2016', fontsize=18)

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')

    return ax


def error_rate_test_mean(df, ax=None, figsize=(10, 7.5), save_as=None):
    """plot both frame error rate and syllable error rate results from
    generating a learning curve

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

    Returns
    -------
    ax : matplotlib.axes.Axes
        either same as axes passed by user, or new axes created
        (when axes is None).s
    """
    if ax is None:
        fig, ax = plt.subplots(2, 2, constrained_layout=True)
        fig.set_size_inches(figsize)

    sns.pointplot(x="train_set_dur", y="frame_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  scale=0.75,
                  ax=ax[0, 0])
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("frame error (%)", fontsize=20)
    ax[0, 0].set_ylim([0., 0.2])
    ax[0, 0].tick_params(labelsize=16)
    ax[0, 0].legend(bbox_to_anchor=(1.01, 1), loc=2,
                 borderaxespad=0.)

    sns.boxplot(x="train_set_dur", y="frame_error_test_mean", data=df, ax=ax[0, 1])
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_ylabel("")
    ax[0, 1].tick_params(labelsize=16)
    ax[0, 1].set_ylim([0., 0.2])

    sns.pointplot(x="train_set_dur", y="syllable_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  ax=ax[1, 0])
    ax[1, 0].set_xlabel("training set duration (s)", fontsize=20)
    ax[1, 0].set_ylabel("syllable error rate\n(distance)", fontsize=20)
    ax[1, 0].set_ylim([0., 1.5])
    ax[1, 0].tick_params(labelsize=16)
    ax[1, 0].legend_.remove()

    sns.boxplot(x="train_set_dur", y="syllable_error_test_mean", data=df, ax=ax[1, 1])
    ax[1, 1].set_xlabel("training set duration (s)", fontsize=20)
    ax[1, 1].set_ylabel("")
    ax[1, 1].set_ylim([0., 1.5])
    ax[1, 1].tick_params(labelsize=16)
    label = 'Koumura\nOkanoya 2016'
    handle = ax[1, 1].scatter([1, 2], [0.84, 0.46], s=75, color='k', marker='s',
                              label=label)
    ax[1, 1].legend([handle], [label])

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')

    return ax
