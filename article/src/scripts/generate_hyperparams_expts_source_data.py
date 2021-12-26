#!/usr/bin/env python
# coding: utf-8
"""generate source data .csv for panel of learning curve figure 
that shows results of running experiments testing hyperparameters"""
import argparse

import pandas as pd
import pyprojroot

PROJROOT = pyprojroot.here()
RESULTS_ROOT = PROJROOT / 'results'


# 'expt' = was run as part of the experiments
# 'learncurve' = was the value we used for the main figure learncurve
HYPERPARAM_EXPTS = {
    'Bengalese_Finches': {
        'hidden_size': [
            (16, 'expt'),
            (64, 'expt'),
            (256, 'learncurve'),
        ],
        'window_size': [
            (22, 'expt'),
            (44, 'expt'),
            (88, 'expt'),
            (176, 'learncurve'),
            (352, 'expt'),
        ],
        'filter_num': [
            (16, 'expt'),
            (32, 'learncurve'),
            (64, 'expt'),
        ],
        'filter_size': [
            (3, 'expt'),
            (5, 'learncurve'),
            (7, 'expt'),
        ],
    },
    'Canaries': {
        'hidden_size': [
            (32, 'expt'),
            (64, 'expt'),
            (512, 'learncurve'),
            (1024, 'expt'),
            (2048, 'expt'),
        ],
        'window_size': [
            (23, 'expt'),
            (46, 'expt'),
            (92, 'expt'),
            (185, 'expt'),
            (370, 'learncurve'),
            (740, 'expt'),
        ]
    }
}

DIRNAME_SPECIES_COL_VAL_MAP = {
    'Bengalese_Finches': 'Bengalese Finch',
    'Canaries': 'Canary',
}


def main(source_data_csv_path):
    dfs = []

    for species, hyperparam_expt_dict in HYPERPARAM_EXPTS.items():
        for hyperparam_expt, params_list in hyperparam_expt_dict.items():

            for param_tuple in params_list:
                param_val, location = param_tuple
                if location == 'expt':
                    param_root = RESULTS_ROOT / species / hyperparam_expt / f'{hyperparam_expt}_{param_val}'
                elif location == 'learncurve':
                    param_root = RESULTS_ROOT / species / 'learncurve'

                results_csv = sorted(param_root.glob('err*.csv'))
                assert len(results_csv) == 1, f'did not find only 1 csv: {results_csv}'
                results_csv = results_csv[0]
                df = pd.read_csv(results_csv)

                # note we convert segment error rate to %
                df.avg_segment_error_rate = df.avg_segment_error_rate * 100

                df['hyperparam_expt'] = hyperparam_expt
                df['hyperparam_val'] = param_val
                df['species'] = species
                dfs.append(df)

    df = pd.concat(dfs)
    df['species'] = df['species'].map(DIRNAME_SPECIES_COL_VAL_MAP)

    df.to_csv(source_data_csv_path,
              index=False)


SOURCE_DATA_CSV_PATH = RESULTS_ROOT / 'hyperparams_expts' / 'source_data.csv'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-data-csv-path',
        default=SOURCE_DATA_CSV_PATH
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(source_data_csv_path=args.source_data_csv_path)
