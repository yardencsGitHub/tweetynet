#!/usr/bin/env python
# coding: utf-8
"""makes error_across_days.csv, source data for figure that plots error across days for Bengalese Finch Song Repository dataset"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyprojroot
import seaborn as sns

import vak


REPO_ROOT = pyprojroot.here()
BFSONGREPO_RESULTS_ROOT = REPO_ROOT.joinpath('results/BFSongRepository').expanduser().resolve()

ERROR_ACROSS_DAYS_CSV_PATH = BFSONGREPO_RESULTS_ROOT.joinpath('error_across_days.csv')

# used below to determine which columns are metrics, and which of those should be converted to error
METRICS = [
    'acc',
    'acc_majority_vote',
    'acc_min_dur_maj_vote',
    'acc_min_segment_dur',
    'levenshtein',
    'levenshtein_majority_vote',
    'levenshtein_min_dur_maj_vote',
    'levenshtein_min_segment_dur',
    'segment_error_rate',
    'segment_error_rate_majority_vote',
    'segment_error_rate_min_dur_maj_vote',
    'segment_error_rate_min_segment_dur',
    'y_pred_np_boundary_err', 
    'y_pred_np_mindur_boundary_err', 
    'y_pred_np_mv_boundary_err', 
    'y_pred_np_mindur_mv_boundary_err',
]


def main():
    csvs = sorted(BFSONGREPO_RESULTS_ROOT.glob('*.csv'))

    # munge all the csvs into one big pandas `DataFrame`, adding columns "bird ID" and "day", where "day" is just 
    n_cols = [f'{name}_n_frame_err' for name in ('y_pred_np', 'y_pred_np_mindur', 'y_pred_np_mv', 'y_pred_np_mindur_mv')] 

    records = defaultdict(list)

    for csv in csvs:
        bird_id = csv.name.split('.')[0]

        df = pd.read_csv(csv)
        for y_pred_name in ('y_pred_np', 'y_pred_np_mindur', 'y_pred_np_mv', 'y_pred_np_mindur_mv'):
            df[f'{y_pred_name}_boundary_err'] = df[f'{y_pred_name}_n_unlabeled_err_within_2_timebin_onoffset'] / df[f'{y_pred_name}_n_frame_err']
        for count, date in enumerate(df['date'].unique()):
            day = count + 1
            records['date'].append(date)
            records['day'].append(day)
            records['bird_id'].append(bird_id)

            df_date = df[df['date'] == date]
            for metric in METRICS:
                mn_metric = df_date[metric].mean()
                records[f'mean_{metric}'].append(mn_metric)

    data = pd.DataFrame.from_records(records)

    # add duration for each data by finding corresponding dataset .csv, computing duration, and adding to column
    duration_col = []
    date_col = []  # use to test that we got the order of duration correct

    for bird_id in data['bird_id'].unique():
        # get eval csv files for this bird id
        eval_dataset_csvs = sorted(BFSONGREPO_RESULTS_ROOT.joinpath(f'{bird_id}/eval').glob('*.csv'))
        # and then get their duration and the date
        duration_date_tuples = []
        for eval_dataset_csv in eval_dataset_csvs:
            df_eval = pd.read_csv(eval_dataset_csv)
            duration = vak.io.dataframe.split_dur(df_eval, 'test')
            date_from_df = Path(df_eval['audio_path'][0]).parents[1].name
            date_from_df = int(date_from_df)  # type for date column is int. Yes I know this is not a great idea
            duration_date_tuples.append(
                (duration, date_from_df)
            )

        # now add duration by finding corresponding date
        for date in data[data['bird_id'] == bird_id]['date'].unique():
            dur_date_tup = [dur_date_tup for dur_date_tup in duration_date_tuples if dur_date_tup[1] == date]
            assert len(dur_date_tup) == 1, 'more than one date matched'
            dur_date_tup = dur_date_tup[0]
            duration_col.append(dur_date_tup[0])
            date_col.append(dur_date_tup[1])

    # test dates are in same order as in DataFrame. If test passes, we can just add duration_col as is.
    date_col = np.asarray(date_col)
    assert np.all(np.equal(date_col, data['date'].values))

    duration_col = np.asarray(duration_col)
    data['test_set_duration'] = duration_col

    data['test_set_duration_min'] = data['test_set_duration'] / 60

    print('mean duration (seconds) of test set:', data['test_set_duration'].mean())
    print('std. dev of test set duration  (seconds) :', data['test_set_duration'].std())

    print('mean duration (minutes) of test set:', data['test_set_duration_min'].mean())
    print('std. dev of test set duration (minutes):', data['test_set_duration_min'].std())


    # convert (framewise) accuracy to error, i.e. add columns with (1 - accuracy).  
    # Multiple columns because of measuring accuracy without and with the transformations applied to the network outputs.
    for metric in METRICS:
        if 'acc' in metric:
            err_name = metric.replace('acc', 'err')
            data[f'mean_{err_name}'] = 1 - data[f'mean_{metric}']


    # Check that we have a sane looking result before plotting
    print(data[data['bird_id'] == 'or60yw70'].head())
    print(f"{data['mean_err'].mean():.3f}")
    print(f"{data['mean_err'].std():.3f}")
    print(f"{data['mean_segment_error_rate'].mean():.3f}")
    print(f"{data['mean_segment_error_rate'].std():.3f}")
    print(f"{data['mean_segment_error_rate_majority_vote'].mean():.3f}")
    print(f"{data['mean_segment_error_rate_majority_vote'].std():.3f}")
    print(f"{data['mean_y_pred_np_mv_boundary_err'].mean():.3f}")
    print(f"{data['mean_y_pred_np_mv_boundary_err'].std():.3f}")

    data.to_csv(
        ERROR_ACROSS_DAYS_CSV_FILENAME
    )


if __name__ == '__main__':
    main()
