#!/usr/bin/env python
# coding: utf-8
"""script that makes source data .csv files for figures
that plot performance of TweetyNet models
on other days from the Bengalese Finch Song Repository dataset
with and without clean-up transformations applied to predictions

Note this script expects directories with the following structure:

configs/Bengalese_Finches/behavior/  # <-- specify as `configs_root` argument when calling this script
├── bl26lb16  # <-- 'animal_id_root' in `main` function below
│   ├── 042012  # <-- 'day_dir' in `main` function below
│   │   └── bl26lb16_eval_042012.toml
│   └── 042112
│       └── bl26lb16_eval_042112.toml
├── gr41rd51
│   ├── 062012
│   │   └── gr41rd51_eval_062012.toml
│   ├── ...
├── gy6or6
│   ├── ...
└── or60yw70
    ├── ...
"""
from argparse import ArgumentParser
import configparser  # used to load 'min_segment_dur.ini'
import json
from pathlib import Path

import pandas as pd
import pyprojroot

from article.eval.eval import eval_with_output_tfms, LearncurveAnnot


def eval_train_dur_with_transforms(eval_csv_path,
                                   previous_run_path,
                                   train_dur_to_use,
                                   min_segment_dur,
                                   split="predict",
                                   to_annot=False,
                                   logger=None):
    from vak import config  # avoid circular imports
    from vak.core.learncurve import train_dur_csv_paths as _train_dur_csv_paths
    from vak.logging import log_or_print

    toml_path = sorted(previous_run_path.glob('*.toml'))
    assert len(toml_path) == 1, f'found more than one .toml config file: {toml_path}'

    toml_path = toml_path[0]

    cfg = config.parse.from_toml_path(toml_path)

    # ---- get all the parameters from the learncurve config that we need
    model_config_map = config.models.map_from_path(toml_path,
                                                   cfg.learncurve.models)
    window_size = cfg.dataloader.window_size
    num_workers = cfg.learncurve.num_workers
    normalize_spectrograms = cfg.learncurve.normalize_spectrograms
    train_set_durs = cfg.learncurve.train_set_durs
    if train_dur_to_use not in train_set_durs:
        raise ValueError(
            f'value specified for train_dur_to_use, {train_dur_to_use}, not in '
            f'train_set_durs ({train_set_durs}), from .toml configuration file:\n'
            f'{toml_path}'
        )

    eval_dfs = []
    if to_annot:
        learncurve_annots = []
    else:
        learncurve_annots = None

    # NB: abusing this function to get results directories;
    # we don't actually need the csv paths, just their parent directories
    # which we use to get checkpoints, SpectScaler, etc., for running eval
    train_dur_csv_paths = _train_dur_csv_paths._dict_from_dir(previous_run_path)
    csv_paths = train_dur_csv_paths[train_dur_to_use]

    for replicate_num, this_train_dur_this_replicate_csv_path in enumerate(csv_paths):
        replicate_num += 1  # so log statements below match replicate nums returned by train_dur_csv_paths
        this_train_dur_this_replicate_results_path = (
            this_train_dur_this_replicate_csv_path.parent
        )

        log_or_print(
            f"Evaluating models from replicate {replicate_num} "
            f"using dataset from .csv file: {this_train_dur_this_replicate_results_path}",
            logger=logger,
            level="info",
        )

        if normalize_spectrograms:
            spect_scaler_path = (
                this_train_dur_this_replicate_results_path.joinpath(
                    "StandardizeSpect"
                )
            )
            log_or_print(
                f"Using spect scaler to normalize: {spect_scaler_path}",
                logger=logger,
                level="info",
            )
        else:
            spect_scaler_path = None

        # ---- have to load labelmap before we can get models
        labelmap_path = this_train_dur_this_replicate_results_path.joinpath(
            "labelmap.json"
        )
        log_or_print(
            f"Using labelmap: {labelmap_path}", logger=logger, level="info"
        )
        with labelmap_path.open("r") as f:
            labelmap = json.load(f)

        for model_name, model_config in model_config_map.items():
            log_or_print(
                f"Evaluating model: {model_name}", logger=logger, level="info"
            )
            results_model_root = (
                this_train_dur_this_replicate_results_path.joinpath(model_name)
            )
            ckpt_root = results_model_root.joinpath("checkpoints")
            ckpt_paths = sorted(ckpt_root.glob("*.pt"))
            if any(["max-val-acc" in str(ckpt_path) for ckpt_path in ckpt_paths]):
                ckpt_paths = [
                    ckpt_path
                    for ckpt_path in ckpt_paths
                    if "max-val-acc" in str(ckpt_path)
                ]
                if len(ckpt_paths) != 1:
                    raise ValueError(
                        f"did not find a single max-val-acc checkpoint path, instead found:\n{ckpt_paths}"
                    )
                ckpt_path = ckpt_paths[0]
            else:
                if len(ckpt_paths) != 1:
                    raise ValueError(
                        f"did not find a single checkpoint path, instead found:\n{ckpt_paths}"
                    )
                ckpt_path = ckpt_paths[0]
            log_or_print(
                f"Using checkpoint: {ckpt_path}", logger=logger, level="info"
            )

            (eval_df,
             annots_by_cleanup) = eval_with_output_tfms(eval_csv_path,
                                                        model_config_map,
                                                        checkpoint_path=ckpt_path,
                                                        labelmap=labelmap,
                                                        window_size=window_size,
                                                        num_workers=num_workers,
                                                        spect_scaler_path=spect_scaler_path,
                                                        min_segment_dur=min_segment_dur,
                                                        split=split,
                                                        to_annot=to_annot)
            eval_df['train_set_dur'] = train_dur_to_use
            eval_df['replicate_num'] = replicate_num
            eval_df['model_name'] = model_name
            eval_dfs.append(eval_df)

            if to_annot:
                learncurve_annots.append(
                    LearncurveAnnot(train_dur=train_dur_to_use,
                                    replicate_num=replicate_num,
                                    annots_by_cleanup=annots_by_cleanup)
                )

    eval_dfs = pd.concat(eval_dfs)
    return eval_dfs, learncurve_annots


def main(configs_root,
         min_segment_dur_ini,
         train_dur_to_use,
         split,
         results_dst,
         csv_filename,
         to_annot=False):
    # import inside function to avoid circular imports
    import crowsetta
    import vak.config.parse

    configs_root = Path(configs_root).expanduser()
    results_dst = Path(results_dst).expanduser()

    animal_ID_roots = sorted(
        [subdir
         for subdir in configs_root.iterdir()
         if subdir.is_dir()]
    )
    animal_ID_day_dirs_map = {}
    for animal_ID_root in animal_ID_roots:
        animal_ID_day_dirs_map[animal_ID_root.name] = sorted([subdir
                                                               for subdir in animal_ID_root.iterdir()
                                                               if subdir.is_dir()])

    # get minimum segment durations to use for clean up. Fail early if they're not there.
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # monkeypatch optionxform so it doesn't lowercase animal IDs
    config.read(Path(min_segment_dur_ini).expanduser().resolve())
    min_segment_durs = {k: float(v) for k, v in config['min_segment_dur'].items()}
    for animal_ID_root in animal_ID_roots:
        if animal_ID_root.name not in min_segment_durs:
            raise ValueError(
                f"could not find a minimum segment duration for animal id: {animal_ID_root.name}. "
                f"Found the following animal IDs in the min_segment_dur.ini file: {min_segment_durs.keys()}"
            )

    all_eval_dfs = []
    if to_annot:
        # make dir we will save annotations in, we actually save the files below
        annot_root = results_dst / 'annotations'
        annot_root.mkdir(exist_ok=True)

    for animal_id, day_dirs in animal_ID_day_dirs_map.items():
        min_segment_dur = min_segment_durs[animal_id]

        for day_dir in day_dirs:
            predict_toml_path = sorted(day_dir.glob('*.toml'))
            assert len(predict_toml_path) == 1, f'did not find single predict toml: {predict_toml_path}'
            predict_toml_path = predict_toml_path[0]
            config = vak.config.parse.from_toml_path(predict_toml_path)

            # use predict csv for eval -- by specifying "predict" split when we call eval
            eval_csv_path = config.eval.csv_path

            # get learncurve results dir used to train models from predict config
            results_dir = [parent
                           # notice we are traversing checkpoint_path parents,
                           # not doing a glob for multiple results dirs
                           for parent in config.eval.checkpoint_path.parents
                           if parent.name.startswith('results_')]
            assert len(results_dir) == 1, f"found more than one results dir in path: {config.eval.checkpoint_path}"
            results_dir = results_dir[0]

            (eval_df,
             learncurve_annots) = eval_train_dur_with_transforms(eval_csv_path=eval_csv_path,
                                                                 previous_run_path=results_dir,
                                                                 train_dur_to_use=train_dur_to_use,
                                                                 min_segment_dur=min_segment_dur,
                                                                 split=split,
                                                                 to_annot=to_annot)
            eval_df['avg_error'] = 1 - eval_df['avg_acc']
            eval_df['animal_id'] = animal_id
            eval_df['day'] = day_dir.name
            all_eval_dfs.append(eval_df)

            if to_annot:
                csv_prefix = f'{animal_id}-run-{results_dir.name}-day-{day_dir.name}-'
                for train_dur, replicate_num, annots_by_cleanup in learncurve_annots:  # unpack named tuple
                    for cleanup_type, annots_list in annots_by_cleanup.items():
                        csv_fname = (csv_prefix +
                                     f'traindur-{train_dur}-replicate-{replicate_num}-cleanup-{cleanup_type}.csv')
                        csv_path = annot_root / csv_fname
                        crowsetta.csv.annot2csv(annots_list, str(csv_path))

    curve_df = pd.concat(all_eval_dfs)
    curve_df['animal_id'] = curve_df['animal_id'].astype('category')
    curve_df['avg_error'] = curve_df['avg_error'] * 100

    # add 'train_set_dur_ind' column that maps train set durations to consecutive integers
    # so we can plot with those integers as the xticks, but then change the xticklabels to the actual values
    # -- this lets us avoid having the labels overlap when the training set durations are close to each other
    # e.g., 30 and 45
    train_set_durs = sorted(curve_df['train_set_dur'].unique())
    dur_int_map = dict(zip(train_set_durs, range(len(train_set_durs))))
    curve_df['train_set_dur_ind'] = curve_df['train_set_dur'].map(dur_int_map)

    csv_path = results_dst.joinpath(csv_filename)
    curve_df.to_csv(csv_path, index=False)


PROJ_ROOT = pyprojroot.here()
CONFIGS_ROOT = PROJ_ROOT / 'data' / 'configs'
BFSONGREPO_BEHAV_CONFIGS_ROOT = CONFIGS_ROOT / 'Bengalese_Finches' / 'behavior'
MIN_SEGMENT_DUR_INI = CONFIGS_ROOT / 'min_segment_dur.ini'
BFSONGREPO_BEHAV_RESULTS_DST = PROJ_ROOT / 'results' / 'Bengalese_Finches' / 'behavior'


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--configs_root',
                        help=('root directory where .toml configuration files are located '
                              'for other days of Bengalese Finch Song Repository dataset, '
                              'that will be used to evaluate behavioral analysis from predicted annotations'),
                        default=BFSONGREPO_BEHAV_CONFIGS_ROOT)
    parser.add_argument('--min_segment_dur_ini',
                        help=('path to .ini file with minimum segment durations '
                              'where option name is '),
                        default=MIN_SEGMENT_DUR_INI)
    parser.add_argument('--csv_filename',
                        help='filename of .csv that will be saved by this script in results_root',
                        default='error_across_birds_with_cleanup.csv')
    parser.add_argument('--train_dur_to_use', type=int, default=600)
    parser.add_argument('--split',
                        help=('split to use when evaluating csv path from .toml configuration file; '
                              'defaults to "predict" because the configuration files are all for prediction'),
                        default='test')
    parser.add_argument('--results_dst',
                        help='directory where results will be saved',
                        default=BFSONGREPO_BEHAV_RESULTS_DST)
    parser.add_argument('--to_annot',
                        help=("if this option is added, predictions will be converted to annotations "
                              "and then saved in an 'annotations' directory in 'results_dst'. "
                              "(If the option is not added, then 'to_annot' defaults to False.)"),
                        action='store_true',
                        )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(configs_root=args.configs_root,
         min_segment_dur_ini=args.min_segment_dur_ini,
         train_dur_to_use=args.train_dur_to_use,
         split=args.split,
         results_dst=args.results_dst,
         csv_filename=args.csv_filename,
         to_annot=args.to_annot)
