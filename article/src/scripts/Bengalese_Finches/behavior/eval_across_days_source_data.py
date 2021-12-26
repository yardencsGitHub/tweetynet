#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import pyprojroot


CSV_FNAME = 'error_across_birds_with_cleanup.csv'
RESULTS_ROOT = pyprojroot.here() / 'results' / 'Bengalese_Finches'

learncurve_csv = RESULTS_ROOT / 'learncurve' / CSV_FNAME
eval_across_days_csv = RESULTS_ROOT / 'behavior' / CSV_FNAME


curve_df = pd.read_csv(learncurve_csv)
eval_df = pd.read_csv(eval_across_days_csv)


# tidy learning curve `DataFrame`
TRAIN_SET_DUR = 600

curve_df = curve_df[curve_df.train_set_dur == TRAIN_SET_DUR]

BFSONGREPO_ANIMAL_IDS = set(eval_df.animal_id.unique().tolist())
curve_df = curve_df[curve_df.animal_id.isin(BFSONGREPO_ANIMAL_IDS)]

curve_df['day_int'] = 1.0  # 'day 1' is test set from day we used to train models


# tidy `DataFrame` from running `eval` on other days from dataset
eval_df['day_int'] = np.nan
for animal_id in eval_df.animal_id.unique():
    eval_df.loc[eval_df.animal_id == animal_id, 'day_int'] = pd.factorize(eval_df.loc[eval_df.animal_id == animal_id, 'day'])[0]

eval_df['day_int'] += 2.0


# concatenate the `DataFrame`s into one
df = pd.concat((curve_df, eval_df))

df['day_int'] = df['day_int'].astype(int)

source_data_csv_path = RESULTS_ROOT / 'behavior' / 'eval-across-days.csv'
df.to_csv(source_data_csv_path)
