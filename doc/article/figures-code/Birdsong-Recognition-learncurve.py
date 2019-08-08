from pathlib import Path

import article

HERE = Path(__file__).parent

config_dir = HERE.joinpath('../../../src/configs/')
config_files = config_dir.glob('*BirdsongRecognition*ini')
config_files = sorted([config_file for config_file in config_files])

data_dir = HERE.joinpath('../../../data/')
test_dirs = data_dir / 'BirdsongRecognition'
test_dirs = test_dirs.glob('Bird*/')
test_dirs = sorted([test_dir for test_dir in test_dirs])

csv_fname = str(HERE.joinpath('../../../results/BirdsongRecognition_test.csv'))


def main():
    df = article.util.make_df(config_files, test_dirs,
                              net_name='TweetyNet', csv_fname=csv_fname,
                              train_set_durs=[60, 120, 480])
    agg_df = article.util.agg_df(df, [60, 120, 480])

    save_as = HERE.joinpath('../figures/fig2-frame-error.png')
    ax_frame_err = article.plot.frame_error_rate_test_mean(agg_df, save_as=save_as)

    save_as = HERE.joinpath('../figures/fig2-syllable-error-rate.png')
    ax_syl_err = article.plot.syllable_error_rate_test_mean(agg_df, save_as=save_as)


if __name__ == '__main__':
    main()



