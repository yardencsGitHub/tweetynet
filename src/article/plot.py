"""functions to create plots, basis for figures in article"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


def frame_error_test_mn_plot(df, save_as=None):
    ax = sns.stripplot(x="train_set_dur", y="frame_error_test_mean", data=df, size=12, alpha=0.75)
    sns.boxplot(x="train_set_dur", y="frame_error_test_mean", data=df, ax=ax, showfliers=False)
#    ax.set_ylabel("frame error, test set")
#    ax.set_xlabel("training set duration (s)")

    ax.set_title('Frame error rate as a function of training set size', fontsize=40)
    ax.set_ylabel('Frame error rate\nas measured on test set', fontsize=32)
    ax.set_xlabel('Training set size: duration in s', fontsize=32)

    if save_as:
        plt.savefig(save_as)

    return ax


def syllable_error_rate(all_results_list):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)
    for el in all_results_list:
        lbl = (el['bird_ID'])
        ax.plot(el['train_set_durs'],
                el['mn_test_syl_err'],
                label=lbl,
                linestyle=':',
                marker='o')

    plt.scatter(120, 0.84, s=75)
    plt.text(75, 0.7, 'Koumura & Okanoya 2016,\n0.84 note error rate\nwith 120s training data', fontsize=20)
    plt.scatter(480, 0.5, s=75)
    plt.text(355, 0.35, 'Koumura & Okanoya 2016,\n0.46 note error rate\nwith 480s training data', fontsize=20)

    plt.legend(fontsize=20, loc='upper right');
    plt.title('Syllable error rate as a function of training set size', fontsize=40)
    plt.xticks(el['train_set_durs'])
    plt.tick_params(axis='both', which='major', labelsize=20, rotation=45)
    plt.ylabel('Syllable error rate\nas measured on test set', fontsize=32)
    plt.xlabel('Training set size: duration in s', fontsize=32);
    plt.tight_layout()
    plt.savefig('syl-error-rate-v-train-set-size.png')