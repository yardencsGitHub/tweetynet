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
        plt.savefig(save_as)

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
        plt.savefig(save_as)

    return ax
