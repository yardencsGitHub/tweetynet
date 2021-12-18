#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict
import json
from pathlib import Path

import crowsetta
import numpy as np
import pandas as pd
import pyprojroot
import toml
from tqdm import tqdm
import vak

import article


# we record ground truth and prediction for all of these
# selected using the notebook 'branch-point-inspection.ipynb'
BRANCH_POINTS = {
    'bl26lb16': [('b', 'b'), ('b', 'c')],
    'gr41rd51': [('e', 'f'), ('e', 'i')],
    'gy6or6': [('e', 'e'), ('e', 'f')],
    'or60yw70': [('d', 'e'), ('d', 'g')],
}

# ---- 1. get transition matrices for ground truth data
behav_configs_root = pyprojroot.here() / 'data/configs/Bengalese_Finches/behavior'

animal_ID_roots = sorted(
    [subdir
     for subdir in behav_configs_root.iterdir()
     if subdir.is_dir()]
)
animal_day_config_map = {}
for animal_ID_root in animal_ID_roots:
    animal_day_config_map[animal_ID_root.name] = sorted([subdir
                                                         for subdir in animal_ID_root.iterdir()
                                                         if subdir.is_dir()])

animal_day_config_map = {
    animal_ID: {
        day_dir.name: sorted(day_dir.glob('*.toml'))[0] for day_dir in day_dirs
    }
    for animal_ID, day_dirs in animal_day_config_map.items()
}

print('getting ground truth transition matrices')
animal_day_transmats = {}
for animal_id, day_config_map in tqdm(animal_day_config_map.items()):
    day_transmats = {}
    for day, config_path in day_config_map.items():
        with config_path.open('r') as fp:
            config = toml.load(fp)
        prep_csv_path = pyprojroot.here() / config['EVAL']['csv_path']
        df = pd.read_csv(prep_csv_path)
        annots = vak.annotation.from_df(df)
        day_transmats[day] = article.bfbehav.sequence.transmat_from_annots(annots, thresh=0.002)
    animal_day_transmats[animal_id] = day_transmats

# ---- 2. get transition matrices from predictions made by each model
# (including the multiple training replicates)
behav_results_annot_root = pyprojroot.here() / 'results/Bengalese_Finches/behavior/annotations'

CLEANUP = 'min_segment_dur_majority_vote'

scribe = crowsetta.Transcriber(format='csv')

print('getting transition matrices from model predictions')
animal_day_pred_transmats = {}
for animal_id, day_config_map in tqdm(animal_day_config_map.items()):
    day_transmats = {}
    for day, _ in day_config_map.items():
        annot_csvs = sorted(behav_results_annot_root.glob(f'{animal_id}*{day}*{CLEANUP}.csv'))
        model_transmat_map = {}
        for replicate_num, annot_csv in enumerate(annot_csvs):
            annots = scribe.from_file(annot_csv)
            model_transmat_map[f'replicate {replicate_num}'] = article.bfbehav.sequence.transmat_from_annots(annots, thresh=0.002)
        day_transmats[day] = model_transmat_map
    animal_day_pred_transmats[animal_id] = day_transmats


# ---- 3. make DataFrame of transition probabilities
# from ground truth and predicted annotations
records = []  # to make into DataFrame

for animal_id, transitions in BRANCH_POINTS.items():
    for transition in transitions:
        days = list(animal_day_transmats[animal_id].keys())
        for day in days:
            matrix = animal_day_transmats[animal_id][day].matrix
            states = animal_day_transmats[animal_id][day].states
            row_ind, col_ind = states.index(transition[0]), states.index(transition[1])
            trans_prob = matrix[row_ind, col_ind]
            records.append(
                {
                    'animal_id': animal_id,
                    'day': day,
                    'transition': transition,
                    'prob': trans_prob,
                    'source': 'ground_truth',
                }
            )

            for replicate_num_str, trans_mat_tuple in animal_day_pred_transmats[animal_id][day].items():
                matrix = trans_mat_tuple.matrix
                states = trans_mat_tuple.states
                row_ind, col_ind = states.index(transition[0]), states.index(transition[1])
                trans_prob = matrix[row_ind, col_ind]
                records.append(
                    {
                        'animal_id': animal_id,
                        'day': day,
                        'transition': transition,
                        'prob': trans_prob,
                        'source': 'model',
                        'replicate_num': int(replicate_num_str.split()[-1]),
                    }
                )

print('saving dataframe of transition probabilities for selected branch points')
df = pd.DataFrame.from_records(records)

RESULTS_ROOT = pyprojroot.here() / 'results' / 'Bengalese_Finches'
source_data_csv_path = RESULTS_ROOT / 'behavior' / 'transition-probabilities.csv'
df.to_csv(source_data_csv_path)

animal_xyerr = {}
print('computing mean / std. dev. for predicted probabilities')
for animal_id, transitions in BRANCH_POINTS.items():

    for transition in transitions:
        animal_xyerr[(animal_id, transition)] = defaultdict(list)

        days = list(animal_day_transmats[animal_id].keys())
        for day in days:
            matrix = animal_day_transmats[animal_id][day].matrix
            states = animal_day_transmats[animal_id][day].states
            row_ind, col_ind = states.index(transition[0]), states.index(transition[1])
            trans_prob = matrix[row_ind, col_ind]

            animal_xyerr[(animal_id, transition)]['x'].append(trans_prob)

            y_vals = []
            for replicate_num_str, trans_mat_tuple in animal_day_pred_transmats[animal_id][day].items():
                matrix = trans_mat_tuple.matrix
                states = trans_mat_tuple.states
                row_ind, col_ind = states.index(transition[0]), states.index(transition[1])
                trans_prob = matrix[row_ind, col_ind]
                y_vals.append(trans_prob)

            animal_xyerr[(animal_id, transition)]['y'].append(np.mean(y_vals))
            animal_xyerr[(animal_id, transition)]['yerr'].append(np.std(y_vals))

# keys can't be tuples when saving to .json, convert to strings
animal_xyerr = {str(k): v for k, v in animal_xyerr.items()}

source_data_json_path = RESULTS_ROOT / 'behavior' / 'transition-probabilities-x-y-plot.json'
with source_data_json_path.open('w') as fp:
    json.dump(animal_xyerr, fp)

