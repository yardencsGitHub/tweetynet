#!/usr/bin/env python
# coding: utf-8
"""tests for ground truth and predictions
whether transition points are stable
using statistical method from:

Variable Sequencing Is Actively Maintained in a Well Learned Motor Skill
Timothy L. Warren, Jonathan D. Charlesworth, Evren C. Tumer, Michael S. Brainard
Journal of Neuroscience 31 October 2012, 32 (44) 15414-15425; DOI: 10.1523/JNEUROSCI.1254-12.2012
"""
from pathlib import Path

import crowsetta
import pandas as pd
import pyprojroot
import toml
from tqdm import tqdm
import vak

import article


# we record ground truth and prediction for all of these
# selected using the notebook 'branch-point-inspection.ipynb'
BRANCH_POINTS_TO_TEST = {
    'bl26lb16': ('b', 'c'),
    'gr41rd51': ('e', 'f'),
    'gy6or6': ('e', 'f'),
    'or60yw70': ('e', 'f'),
}

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

print('getting transition matrices from ground truth data')
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


behav_results_annot_root = pyprojroot.here() / 'results/Bengalese_Finches/behavior/annotations'

CLEANUP = 'min_segment_dur_majority_vote'

scribe = crowsetta.Transcriber(format='csv')

print('getting transition matrices from predicted annotations')
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

records = []  # to make dataframe that we dump out
for animal_id, branch_point in BRANCH_POINTS_TO_TEST.items():
    print(f'running test for: {animal_id}')
    days = list(animal_day_transmats[animal_id].keys())
    for day in days:
        print(f'for day: {day}')
        p_vals, alpha = article.bfbehav.stats.perm_test_across_models(animal_day_transmats_true=animal_day_transmats,
                                                                       animal_day_transmats_pred=animal_day_pred_transmats,
                                                                       animal_id=animal_id,
                                                                       day=day,
                                                                       from_state=branch_point[0],
                                                                       transition=branch_point,
                                                                       n_perm=1000,)
        for p_val in p_vals:
            records.append(
                {
                    'animal_id': animal_id,
                    'branch_point': str(branch_point),
                    'day': day,
                    'p_val': p_val,
                    'alpha': alpha,
                }
            )

df = pd.DataFrame.from_records(records)
CSV_FNAME = 'branch_points_stats.csv'
RESULTS_ROOT = pyprojroot.here() / 'results' / 'Bengalese_Finches' / 'behavior'
df.to_csv(RESULTS_ROOT / CSV_FNAME)
