#!/usr/bin/env python
# coding: utf-8
"""makes error_across_birds.csv, source data for figure that plots error across birds for Birdsong Recognition dataset"""
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyprojroot
import seaborn as sns

import tqdm

re_int = re.compile(r'[0-9]+')

def int_from_dir_path(dir_path):
    name = dir_path.name
    return int(re_int.search(name)[0])

REPO_ROOT = pyprojroot.here()
BR_RESULTS_ROOT = REPO_ROOT.joinpath('results/BirdsongRecognition')

ERROR_ACROSS_BIRDS_CSV_PATH = BR_RESULTS_ROOT.joinpath('error_across_birds.csv')


def main():
    BIRDS_ROOTS = sorted(BR_RESULTS_ROOT.glob('Bird*'))

    dfs = []

    for bird_root in BIRDS_ROOTS:
        if bird_root.name == 'Bird10':
            continue  # didn't run experiment, not enough data for test set of correct duration
        else:
            bird_num = int_from_dir_path(bird_root)
            results_roots = sorted(bird_root.glob('results_*'))
            most_recent_results = results_roots[-1]
            df = pd.read_csv(most_recent_results.joinpath('learning_curve.csv'))
            df['avg_error'] = 1 - df['avg_acc']
            df['bird'] = bird_num
            dfs.append(df)

    curve_df = pd.concat(dfs)
    # make 'bird' the first column
    columns = ['bird', 'train_set_dur', 'replicate_num', 'model_name', 'avg_acc', 'avg_levenshtein', 'avg_loss', 'avg_segment_error_rate', 'avg_error']
    curve_df = curve_df[columns]
    curve_df['bird'] = curve_df['bird'].astype('category')

    curve_df['avg_error'] = curve_df['avg_error'] * 100

    # add 'train_set_dur_ind' column that maps train set durations to consecutive integers
    # so we can plot with those integers as the xticks, but then change the xticklabels to the actual values
    # -- this lets us avoid having the labels overlap when the training set durations are close to each other
    # e.g., 30 and 45
    train_set_durs = sorted(curve_df['train_set_dur'].unique())
    dur_int_map = dict(zip(train_set_durs, range(len(train_set_durs))))
    curve_df['train_set_dur_ind'] = curve_df['train_set_dur'].map(dur_int_map)

    curve_df.head()

    TRAIN_DUR_IND_MAP = {
        k:v for k, v in zip(
            sorted(curve_df['train_set_dur'].unique()), 
            sorted(curve_df['train_set_dur_ind'].unique())
        )
    }
    
    curve_df.to_csv(ERROR_ACROSS_BIRDS_CSV_PATH)


if __name__ == '__main__':
    main()
