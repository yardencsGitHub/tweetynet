"""functions to create plots, basis for figures in article"""
import matplotlib.pyplot as plt
import seaborn as sns


def frame_error_rate(all_results_list):
    all_mn_test_err = []
    for el in all_results_list:
        all_mn_test_err.append(el['mn_test_err'])
    all_mn_test_err = np.asarray(all_mn_test_err)

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    for el in all_results_list:
        lbl = (el['bird_ID'])
        ax.plot(el['train_set_durs'],
                el['mn_test_err'],
                label=lbl,
                linestyle='--',
                marker='o')
    ax.plot(el['train_set_durs'], np.median(all_mn_test_err, axis=0),
            linestyle='--', marker='o', linewidth=3, color='k', label='median across birds')

    fig.set_size_inches(16, 8)
    plt.legend(fontsize=20)
    plt.xticks(el['train_set_durs'])
    plt.tick_params(axis='both', which='major', labelsize=20, rotation=45)
    plt.title('Frame error rate as a function of training set size', fontsize=40)
    plt.ylabel('Frame error rate\nas measured on test set', fontsize=32)
    plt.xlabel('Training set size: duration in s', fontsize=32);
    plt.tight_layout()
    plt.savefig('frame-err-rate-v-train-set-size.png')


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