#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import pyprojroot
import scipy.stats


def convert_seg_error_rate_pct(df):
    df.avg_segment_error_rate = df.avg_segment_error_rate * 100
    return df


RESULTS_ROOT = pyprojroot.here() / 'results'

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

records = []

for species in all_df['Species'].unique():
    species_df = all_df[all_df.Species == species]
    for train_dur in species_df.train_set_dur.unique():
        for segmentation in ('manually-cleaned', 'semi-automated-cleaning', 'not-cleaned'):
            # `x` and `y` are the argument names for `scipy.stats.wilcoxon`
            x = species_df[
                (species_df['Model'] == 'SVM') & 
                (species_df['segmentation'] == segmentation) & 
                (species_df['Species'] == species) &
                (species_df['train_set_dur'] == train_dur)
            ].avg_segment_error_rate.values
            
            y = species_df[
                (species_df['Model'] == 'TweetyNet') & 
                (species_df['train_set_dur'] == train_dur)
            ].avg_segment_error_rate.values
            
            wilcoxon_results = scipy.stats.wilcoxon(x,y)
            levene_results = scipy.stats.levene(x,y)
            
            records.append(
                {
                    'species': species,
                    'train_dur': train_dur,
                    'segmentation': segmentation,
                    'mean_diff': np.mean(y - x),
                    'std_diff': np.std(y - x),
                    'pval_wilcoxon': wilcoxon_results.pvalue,
                    'pval_levene': levene_results.pvalue,
                }
            )

df = pd.DataFrame.from_records(records)

df.to_csv(RESULTS_ROOT / 'tweetynet_vs_svm' / 'stats.csv')
