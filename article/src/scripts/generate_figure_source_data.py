#!/usr/bin/env python
# coding: utf-8
import shutil

import pandas as pd
import pyprojroot


def convert_seg_error_rate_pct(df):
    df.avg_segment_error_rate = df.avg_segment_error_rate * 100
    return df


def generate_fig3_source_data():
    RESULTS_ROOT = pyprojroot.here() / 'results'

    FIG_ROOT = pyprojroot.here() / 'doc' / 'figures' / 'mainfig_tweetynet_v_svm'
    FIG_ROOT.mkdir(exist_ok=True)

    segmentation_map = {
        'ground_truth': 'segmented audio, manually cleaned',
        'resegment': 'segmented audio, not cleaned',
        'semi-automated-cleaning': 'segmented audio, semi-automated cleaning',
        'not-cleaned': 'segmented audio, not cleaned',
        'manually-cleaned': 'segmented audio, manually cleaned'
    }
    hvc_dfs = []

    csv_filename = 'segment_error_across_birds.hvc.csv'

    for species in ('Bengalese_Finches', 'Canaries'):
        species_csv = RESULTS_ROOT / f'{species}/hvc/{csv_filename}'
        df = pd.read_csv(species_csv)

        df['Model'] = 'SVM'
        df['Input to model'] = df['segmentation'].map(segmentation_map)
        df['Species'] = species
        hvc_dfs.append(df)

    hvc_df = pd.concat(hvc_dfs)

    curve_df = []

    for species in ('Bengalese_Finches', 'Canaries'):
        LEARNCURVE_RESULTS_ROOT = pyprojroot.here() / 'results' / species / 'learncurve'
        error_csv_path = LEARNCURVE_RESULTS_ROOT.joinpath('error_across_birds_with_cleanup.csv')
        df = pd.read_csv(error_csv_path)

        df = df[df.animal_id.isin(hvc_df.animal_id.unique())]
        df['Model'] = 'TweetyNet'
        df['Input to model'] = 'spectrogram'
        df['Species'] = species
        curve_df.append(df)

    del df
    curve_df = pd.concat(curve_df)

    CLEANUP = 'min_segment_dur_majority_vote'

    curve_df = curve_df[
        curve_df.cleanup == CLEANUP
        ]

    all_df = pd.concat([hvc_df, curve_df])

    all_df = convert_seg_error_rate_pct(all_df)

    gb = all_df.groupby(by=['Species', 'Model', 'Input to model', 'animal_id', 'train_set_dur'])
    df_agg = gb.agg(
        mean_seg_err = pd.NamedAgg('avg_segment_error_rate', 'mean'),
        median_seg_err = pd.NamedAgg('avg_segment_error_rate', 'median'),
        std_seg_err = pd.NamedAgg('avg_segment_error_rate', 'std')
    )

    data = df_agg.reset_index()  # ``data`` DataFrame for use with ``seaborn``

    data.to_csv(FIG_ROOT / 'fig3-data1.csv')


def filter_cleanups(df, cleanups):
    return df[df.cleanup.isin(cleanups)]


def clean_df(df, species, cleanups):
    df = convert_seg_error_rate_pct(df)
    df = add_species(df, species)
    df = filter_cleanups(df, cleanups)
    return df


def add_species(df, species):
    df['species'] = species
    return df


def generate_fig4_source_data():
    PROJ_ROOT = pyprojroot.here()
    RESULTS_ROOT = PROJ_ROOT / 'results'
    BF_RESULTS_ROOT = RESULTS_ROOT / 'Bengalese_Finches' / 'learncurve'
    CANARY_RESULTS_ROOT = RESULTS_ROOT / 'Canaries' / 'learncurve'
    FIGS_ROOT  = PROJ_ROOT / 'doc' / 'figures'

    THIS_FIG_ROOT = fname = FIGS_ROOT / 'mainfig_across_individuals_species'
    THIS_FIG_ROOT.mkdir(exist_ok=True)

    CLEANUPS = (
        'none',
        'min_segment_dur_majority_vote'
    )

    bf_error_csv_path = BF_RESULTS_ROOT.joinpath('error_across_birds_with_cleanup.csv')
    bf_curve_df = pd.read_csv(bf_error_csv_path)

    bf_curve_df = clean_df(
        bf_curve_df,
        'Bengalese Finch',
        CLEANUPS
    )

    canary_error_csv_path = CANARY_RESULTS_ROOT.joinpath('error_across_birds_with_cleanup.csv')
    canary_curve_df = pd.read_csv(canary_error_csv_path)

    canary_curve_df = clean_df(
        canary_curve_df,
        'Canary',
        CLEANUPS
    )

    for data_num, df in enumerate((bf_curve_df, canary_curve_df)):
        df.to_csv(
            THIS_FIG_ROOT / f'fig4-data{data_num + 1}.csv'
        )


def generate_fig5_source_data():
    PROJ_ROOT = pyprojroot.here()
    RESULTS_ROOT = PROJ_ROOT / 'results'
    BF_RESULTS_ROOT = RESULTS_ROOT / 'Bengalese_Finches' / 'learncurve'
    CANARY_RESULTS_ROOT = RESULTS_ROOT / 'Canaries' / 'learncurve'
    FIGS_ROOT  = PROJ_ROOT / 'doc' / 'figures'

    THIS_FIG_ROOT = fname = FIGS_ROOT / 'mainfig_postprocess_error_rates'
    THIS_FIG_ROOT.mkdir(exist_ok=True)

    # column name is "cleanup" but in the paper we use the term "post-processing"
    # to avoid confusion with where we refer to "clean ups" of other models (e.g. SVM)
    CLEANUPS = (
        'none',
        'min_segment_dur_majority_vote'
    )

    # so we'll add a column 'post-processing' that maps cleanups --> with/without post-process
    POST_PROCESS_MAP = {
        'none': 'without',
        'min_segment_dur_majority_vote': 'with',
    }

    bf_error_csv_path = BF_RESULTS_ROOT.joinpath('error_across_birds_with_cleanup.csv')
    bf_curve_df = pd.read_csv(bf_error_csv_path)

    bf_curve_df = clean_df(
        bf_curve_df,
        'Bengalese Finches',
        CLEANUPS
    )

    canary_error_csv_path = CANARY_RESULTS_ROOT.joinpath('error_across_birds_with_cleanup.csv')
    canary_curve_df = pd.read_csv(canary_error_csv_path)

    canary_curve_df = clean_df(
        canary_curve_df,
        'Canaries',
        CLEANUPS
    )

    # only plot canaries mean for training set durations where we have results for all birds, which is > 180
    canary_curve_df = canary_curve_df[canary_curve_df.train_set_dur > 180]

    curve_df = pd.concat((bf_curve_df, canary_curve_df))
    curve_df = curve_df.rename(columns={'species': 'Species'})  # so it's capitalized in figure, legend, etc.
    curve_df['Post-processing'] = curve_df['cleanup'].map(POST_PROCESS_MAP)
    train_set_durs = sorted(curve_df['train_set_dur'].unique())
    dur_int_map = dict(zip(train_set_durs, range(len(train_set_durs))))
    curve_df['train_set_dur_ind'] = curve_df['train_set_dur'].map(dur_int_map)

    hyperparams_expt_csv_path = RESULTS_ROOT / 'hyperparams_expts' / 'source_data.csv'
    hyperparams_expt_df = pd.read_csv(hyperparams_expt_csv_path)

    hyperparams_expt_df = filter_cleanups(hyperparams_expt_df,
                                          CLEANUPS)

    hyperparams_expt_df['Post-processing'] = hyperparams_expt_df['cleanup'].map(POST_PROCESS_MAP)

    curve_df.to_csv(THIS_FIG_ROOT / 'fig5-data1.csv')
    hyperparams_expt_df.to_csv(THIS_FIG_ROOT / 'fig5-data2.csv')


def generate_fig6_source_data():
    RESULTS_ROOT = pyprojroot.here()  / 'results' / 'Bengalese_Finches' / 'behavior'

    FIG_ROOT = pyprojroot.here() / 'doc' / 'figures' / 'mainfig_bf_behavior'


    EVAL_CSV_FNAME = 'eval-across-days.csv'
    eval_across_days_csv = RESULTS_ROOT / EVAL_CSV_FNAME

    eval_df = pd.read_csv(eval_across_days_csv)

    # note we convert segment error rate to %
    eval_df.avg_segment_error_rate = eval_df.avg_segment_error_rate * 100

    df_minsegdur_majvote = eval_df[eval_df.cleanup == 'min_segment_dur_majority_vote']
    df_minsegdur_majvote.to_csv(FIG_ROOT / 'fig6-data1.csv')


    TRANS_PROBS_CSV_FNAME = 'transition-probabilities.csv'
    probs_csv = RESULTS_ROOT / TRANS_PROBS_CSV_FNAME
    df = pd.read_csv(probs_csv)
    df_gr41rd51 = df[df.animal_id == 'gr41rd51'].copy()
    df_gr41rd51['Source'] = df_gr41rd51.source.map({'ground_truth': 'Ground truth', 'model': 'Model predictions'})
    df_gr41rd51.to_csv(FIG_ROOT / 'fig6-data2.csv')

    JSON_FNAME = 'transition-probabilities-x-y-plot.json'
    xyerr_json = RESULTS_ROOT / JSON_FNAME
    shutil.copy(xyerr_json, FIG_ROOT / 'fig6-data3.json')


if __name__ == '__main__':
    generate_fig3_source_data()
    generate_fig4_source_data()
    generate_fig5_source_data()
    generate_fig6_source_data()
