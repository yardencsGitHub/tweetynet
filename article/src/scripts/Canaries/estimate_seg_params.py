"""estimate "least-bad" audio segmenting parameters from canaries, 
using statistics of segments in the manually-cleaned ground truth data

estimates amplitude threshold as the median amplitude at onsets and offsets, 
and estimates minimum segment duration and silent interval as
the 10th quantile of an array of those values.
Computed separately for each individual bird's song.
"""
import argparse
import configparser
from pathlib import Path

import numpy as np
import pandas as pd
import pyprojroot


PROJ_ROOT = pyprojroot.here()
SEGMENT_PARAMS_INI = PROJ_ROOT / 'data' / 'configs' / 'segment_params.ini'


SEGMENTING_PARAMS_FEATURE_MAP = {
    'min_segment_dur': 'segment_dur',
    'min_silent_interval': 'silent_interval',
    'threshold': 'amplitude'
}

SEG_FEATURE_CSVS_ROOT = pyprojroot.here() / 'results/Canaries/seg_stats'


def main(seg_feature_csvs_root=SEG_FEATURE_CSVS_ROOT,
         seg_params_ini=SEGMENT_PARAMS_INI,
         canary_ids=('llb11', 'llb16', 'llb3'),):
    seg_feature_csvs_root = Path(seg_feature_csvs_root)

    seg_feature_dfs = {
        feature: pd.read_csv(seg_feature_csvs_root / f'{feature}.csv')
        for feature in SEGMENTING_PARAMS_FEATURE_MAP.values()
    }

    seg_feature_dfs = {
        feature: df[df[feature] > 0.]  # filter out e..g spurious negative values
        if feature != 'amplitude' else df
        for feature, df in seg_feature_dfs.items()

    }

    QUANTILE = {
        'min_segment_dur': 0.05,
        'min_silent_interval': 0.005,
        'threshold': 0.5,
    }

    seg_params_by_canary_id = {
        canary_id: {}
        for canary_id in canary_ids
    }

    for seg_param, seg_feature in SEGMENTING_PARAMS_FEATURE_MAP.items():
        feature_df = seg_feature_dfs[seg_feature]

        for canary_id in canary_ids:
            canary_df = feature_df[feature_df.animal_id == canary_id]

            param_val = np.quantile(
                a=canary_df[seg_feature].values, q=QUANTILE[seg_param]
            )
            if seg_param != 'threshold':
                # round min seg dur / silent interval to 3 decimal places, i.e. milliseconds
                param_val = round(param_val, 3)

            seg_params_by_canary_id[canary_id][seg_param] = param_val

    config_parser = configparser.ConfigParser()
    if seg_params_ini.exists():
        config_parser.read(seg_params_ini)
    for canary_id, params in seg_params_by_canary_id.items():
        for param_name, param_val in params.items():
            if not config_parser.has_section(param_name):
                config_parser.add_section(param_name)
            config_parser[param_name][canary_id] = str(param_val)
    with seg_params_ini.open('w') as fp:
        config_parser.write(fp)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-feature-csvs-root',
                        help=('path to root of directory containing .csv files '
                              f'saved by "generate_segment_feature_vals_csvs.py" script, '
                              f'default is: {SEG_FEATURE_CSVS_ROOT}'),
                        default=SEG_FEATURE_CSVS_ROOT)

    parser.add_argument(
        '--canary-ids',
        nargs='+',
        default=('llb11', 'llb16', 'llb3')
    )
    parser.add_argument('--seg-params-ini',
                        help=("path to .ini file with segmenting parameters "
                              "for audio files from each animal"),
                        default=SEGMENT_PARAMS_INI
                        )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(seg_feature_csvs_root=args.seg_feature_csvs_root,
         seg_params_ini=args.seg_params_ini,
         canary_ids=args.canary_ids)
