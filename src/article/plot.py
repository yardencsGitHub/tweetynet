"""functions to create plots, basis for figures in article"""
import matplotlib.pyplot as plt
import seaborn as sns


def frame_error_rate_test_mean(df, ax=None, figsize=(10, 7.5), save_as=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
    sns.pointplot(x="train_set_dur", y="frame_error_test_mean",
                  hue="animal_ID", data=df,
                  linestyles=['--' for _ in df.animal_ID.unique()],
                  ax=ax)

    mean_fet_mn = df.groupby('train_set_dur')['frame_error_test_mean'].mean()
    with plt.rc_context({'lines.linewidth': 2}):
        mn_plot = sns.pointplot(mean_fet_mn.index,
                                mean_fet_mn.values,
                                color='black', ax=ax)

    ax.set_ylabel("frame error,\ntest set", fontsize=28)
    ax.set_xlabel("training set duration (s)", fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2,
              borderaxespad=0., fontsize=20)

    max_zorder = max([_.zorder for _ in ax.get_children()])
    mn_plot.set_zorder(max_zorder + 1)

    if save_as:
        plt.savefig(save_as)

    return ax


def syllable_error_rate_test_mean(df, ax=None, figsize=(10, 7.5), save_as=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    sns.pointplot(x="train_set_dur", y="syllable_error_test_mean",
                hue="animal_ID", data=df,
                linestyles=['--' for _ in df.animal_ID.unique()],
                ax=ax)

    mean_sert_mn = df.groupby('train_set_dur')['syllable_error_test_mean'].mean()
    with plt.rc_context({'lines.linewidth': 2}):
        mn_plot = sns.pointplot(mean_sert_mn.index,
                                mean_sert_mn.values,
                                color='black', ax=ax)

    ax.set_ylabel("syllable error rate,\ntest set", fontsize=28)
    ax.set_xlabel("training set duration (s)", fontsize=28)
    ax.tick_params(labelsize=20)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2,
              borderaxespad=0., fontsize=20)

    max_zorder = max([_.zorder for _ in ax.get_children()])
    mn_plot.set_zorder(max_zorder + 1)

    if save_as:
        plt.savefig(save_as)

    return ax
