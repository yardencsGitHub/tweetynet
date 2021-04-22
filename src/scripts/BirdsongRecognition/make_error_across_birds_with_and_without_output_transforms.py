#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser

from collections import defaultdict
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import pyprojroot
import torch
from tqdm import tqdm

from vak import config, io, models, transforms
from vak.datasets.vocal_dataset import VocalDataset
import vak.device
import vak.files
from vak.labeled_timebins import lbl_tb2segments, majority_vote_transform, lbl_tb_segment_inds_list,     remove_short_segments
from vak.core.learncurve import train_dur_csv_paths as _train_dur_csv_paths
from vak.logging import log_or_print


def compute_metrics(metrics, 
                    y_true, 
                    y_pred,
                    y_true_labels,
                    y_pred_labels):
    """helper function to compute metrics

    Parameters
    ----------
    metrics : dict
        where keys are metric names and values are callables that compute the metric
        given ground truth and prediction
    y_true : torch.Tensor
        vector of labeled time bins
    y_pred : torch.Tensor
        vector of labeled time bins
    y_true_labels : str
        sequence of segment labels
    y_pred_labels : str
        sequence of segment labels

    Returns
    -------
    metric_vals : defaultdict
    """
    metric_vals = {}

    for metric_name, metric_callable in metrics.items():
        if metric_name == 'acc':
            metric_vals[metric_name] = metric_callable(y_pred, y_true)
        elif metric_name == 'levenshtein':
            metric_vals[metric_name] = metric_callable(y_pred_labels, y_true_labels)
        elif metric_name == 'segment_error_rate':
            metric_vals[metric_name] = metric_callable(y_pred_labels, y_true_labels)

    return metric_vals


def boundary_err(y_pred, 
                 y_true, 
                 t,
                 onsets_s,
                 offsets_s,
                 timebin_dur,
                 n_timebin_from_onoffset,
                 unlabeled_class=0):
    frame_err_vec = y_true != y_pred
    n_frame_err = int(frame_err_vec.sum().item())
    unlabeled_err = np.logical_and(
        frame_err_vec,
        np.logical_or(y_true == unlabeled_class, y_pred == unlabeled_class)
    )
    t_unlabeled_err = t[unlabeled_err]
    t_unlabeled_err_from_onset_offset = [
        min(np.abs(np.concatenate((onsets_s, offsets_s)) - a_time))
        for a_time in t_unlabeled_err
    ]
    counts, _ = np.histogram(t_unlabeled_err_from_onset_offset, bins=np.arange(0.0, 1.0, timebin_dur))
    n_unlabeled_err_within_n_timebin = counts[:n_timebin_from_onoffset].sum()
    return n_unlabeled_err_within_n_timebin / n_frame_err


# number of timebins from an onset or offset
# in which we count errors involving "unlabeled" / silent gaps
N_TIMEBINS_FROM_ONOFFSET = 2


def eval_with_output_tfms(csv_path,
                          model_config_map,
                          checkpoint_path,
                          labelmap,
                          window_size,
                          num_workers,
                          n_timebin_from_onoffset=N_TIMEBINS_FROM_ONOFFSET,
                          split="test",
                          spect_scaler_path=None,
                          device='cuda',
                          spect_key='s',
                          timebins_key='t',
                          logger=None):
    """computes evaluation metrics on a dataset

    computes the metrics without and with "majority vote" transform

    Returns
    -------
    df : pandas.Dataframe
    """
    if spect_scaler_path:
        log_or_print(
            f"loading spect scaler from path: {spect_scaler_path}",
            logger=logger,
            level="info",
        )
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        log_or_print(
            f"not using a spect scaler",
            logger=logger,
            level="info",
        )
        spect_standardizer = None

    # ---- make eval dataset that we'll use to compute metrics
    # each batch will give us dict with 'spect', 'annot' and 'spect_path'
    # we can use 'spect_path' to find prediction in pred_dict and then compare to target
    # dict also includes 'padding_mask' so we can "unpad" the prediction vectors
    item_transform = transforms.get_defaults('eval',
                                             spect_standardizer,
                                             window_size=window_size,
                                             return_padding_mask=True,
                                             )

    eval_dataset = VocalDataset.from_csv(csv_path=csv_path,
                                         split=split,
                                         labelmap=labelmap,
                                         spect_key=spect_key,
                                         timebins_key=timebins_key,
                                         item_transform=item_transform,
                                         )

    eval_data = torch.utils.data.DataLoader(dataset=eval_dataset,
                                            shuffle=False,
                                            # batch size 1 because each spectrogram reshaped into a batch of windows
                                            batch_size=1,
                                            num_workers=num_workers)

    df = pd.read_csv(csv_path)  # load because we use below for spect_paths as well
    # get timebin dur to use when converting labeled timebins to labels, onsets and offsets
    timebin_dur = io.dataframe.validate_and_get_timebin_dur(df)

    # ---- make pred dataset to actually do predictions
    # will call model.predict() and get back dict with predictions and spect paths
    pred_transform = transforms.get_defaults('predict',
                                             spect_standardizer,
                                             window_size=window_size,
                                             return_padding_mask=False,
                                             )

    # can't use classmethod because we use 'test' split instead of 'predict' with no annotation
    # so we instantiate directly
    df_split = df[df['split'] == split]
    pred_dataset = VocalDataset(csv_path=csv_path,
                                spect_paths=df_split['spect_path'].values,
                                annots=None,
                                labelmap=labelmap,
                                spect_key=spect_key,
                                timebins_key=timebins_key,
                                item_transform=pred_transform,
                                )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            batch_size=1,  # hard coding to make this work for now
                                            num_workers=num_workers)

    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    if device is None:
        device = vak.device.get_default_device()

    records = defaultdict(list)  # will be used with pandas.DataFrame.from_records to make output csv
    to_long_tensor = transforms.ToLongTensor()

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
            )

    for model_name, model in models_map.items():
        model.load(checkpoint_path)
        metrics = model.metrics  # metric name -> callable map we use below in loop

        pred_dict = model.predict(pred_data=pred_data,
                                  device=device)

        progress_bar = tqdm(eval_data)
        for ind, batch in enumerate(progress_bar):
            for cleanup_type in ('none', 'majority vote'):
                # append this first at beginning of code block
                records['cleanup'].append(cleanup_type)
                
                y_true, padding_mask, spect_path = batch['annot'], batch['padding_mask'], batch['spect_path']
                if isinstance(spect_path, list) and len(spect_path) == 1:
                    spect_path = spect_path[0]  # __getitem__ returns 1-element list for paths (with batch size 1)
                t = vak.files.spect.load(spect_path)[timebins_key]
                records['spect_path'].append(spect_path)  # remove str from tuple
                y_true = y_true.to(device)
                y_true_np = np.squeeze(y_true.cpu().numpy())
                y_true_labels, onsets_s, offsets_s = lbl_tb2segments(y_true_np,
                                                                     labelmap=labelmap,
                                                                     t=t)
                y_true_labels = ''.join(y_true_labels.tolist())

                y_pred = pred_dict[spect_path]
                y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
                y_pred = torch.flatten(y_pred)
                y_pred = y_pred.unsqueeze(0)[padding_mask]
                y_pred_np = np.squeeze(y_pred.cpu().numpy())

                # --- apply cleanup transform to output if any
                if cleanup_type == 'none':
                    y_pred_labels, _, _ = lbl_tb2segments(y_pred_np,
                                          labelmap=labelmap,
                                          t=t,
                                          min_segment_dur=None,
                                          majority_vote=False)
                    y_pred_labels = ''.join(y_pred_labels.tolist())

                elif cleanup_type == 'majority vote':
                    # need segment_inds_list for transfom
                    segment_inds_list = lbl_tb_segment_inds_list(y_pred_np,
                                                                 unlabeled_label=labelmap['unlabeled'])

                    # ---- majority vote transform
                    y_pred_np = majority_vote_transform(y_pred_np, segment_inds_list)
                    y_pred = to_long_tensor(y_pred_np).to(device)
                    y_pred_labels, _, _ = lbl_tb2segments(y_pred_np,
                                                          labelmap=labelmap,
                                                          t=t,
                                                          min_segment_dur=None,
                                                          majority_vote=False)
                    y_pred_labels = ''.join(y_pred_labels.tolist())

                metric_vals_batch = compute_metrics(metrics, y_true, y_pred, y_true_labels, y_pred_labels)
                for metric_name, metric_val in metric_vals_batch.items():
                    records[metric_name].append(metric_val)

                # here we measure the number of times a frame error occurs involving 'unlabeled' timebins
                # within some fixed number of timebins from an onset or offset.
                # the idea being that if those specific frame errors account
                # for a large number of all frame errors,
                # then most frame errors are due to noisiness of segmentation
                bnd_err = boundary_err(y_pred_np, 
                                       y_true_np, 
                                       t,
                                       onsets_s,
                                       offsets_s,
                                       timebin_dur,
                                       n_timebin_from_onoffset,
                                       unlabeled_class=labelmap['unlabeled'])
                # this is the same row in records as 'metric_name` and 'cleanup_type' above
                records['pct_boundary_err'].append(bnd_err)

        eval_df = pd.DataFrame.from_records(records)
        gb = eval_df.groupby('cleanup').agg('mean')
        gb = gb.add_prefix('avg_')
        eval_df = gb.reset_index()

    return eval_df


def from_previous_run_path(previous_run_path,
                           logger=None):
    previous_run_path = Path(previous_run_path)
    toml_path = sorted(previous_run_path.glob('*.toml'))
    assert len(toml_path) == 1, f'found more than one .toml config file: {toml_path}'
    toml_path = toml_path[0]

    cfg = config.parse.from_toml_path(toml_path)

    # ---- get all the parameters from the config we need
    model_config_map = config.models.map_from_path(toml_path, cfg.learncurve.models)
    csv_path = cfg.learncurve.csv_path
    window_size = cfg.dataloader.window_size
    num_workers = cfg.learncurve.num_workers
    normalize_spectrograms = cfg.learncurve.normalize_spectrograms

    log_or_print(
        f"Loading previous training subsets from:\n{previous_run_path}",
        logger=logger,
        level="info",
    )

    train_dur_csv_paths = _train_dur_csv_paths._dict_from_dir(previous_run_path)

    eval_dfs = []
    for train_dur, csv_paths in train_dur_csv_paths.items():
        for replicate_num, this_train_dur_this_replicate_csv_path in enumerate(
            csv_paths
        ):
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

                eval_df = eval_with_output_tfms(csv_path,
                                                model_config_map,
                                                checkpoint_path=ckpt_path,
                                                labelmap=labelmap,
                                                window_size=window_size,
                                                num_workers=num_workers,
                                                spect_scaler_path=spect_scaler_path)
                eval_df['train_set_dur'] = train_dur
                eval_df['replicate_num'] = replicate_num
                eval_df['model_name'] = model_name
                eval_dfs.append(eval_df)

    eval_dfs = pd.concat(eval_dfs)
    return eval_dfs


re_int = re.compile(r'[0-9]+')


def int_from_dir_path(dir_path):
    name = dir_path.name
    return int(re_int.search(name)[0])


def main(results_root,
         csv_filename):
    results_root = Path(results_root)
    birds_roots = sorted(results_root.glob('Bird*'))

    all_eval_dfs = []

    for bird_root in birds_roots:
        bird_num = int_from_dir_path(bird_root)
        results_roots = sorted(bird_root.glob('results_*'))
        most_recent_results = results_roots[-1]
        eval_dfs = from_previous_run_path(previous_run_path=most_recent_results)
        eval_dfs['avg_error'] = 1 - eval_dfs['avg_acc']
        eval_dfs['bird'] = bird_num
        all_eval_dfs.append(eval_dfs)

    curve_df = pd.concat(all_eval_dfs)
    curve_df['bird'] = curve_df['bird'].astype('category')
    curve_df['avg_error'] = curve_df['avg_error'] * 100

    # add 'train_set_dur_ind' column that maps train set durations to consecutive integers
    # so we can plot with those integers as the xticks, but then change the xticklabels to the actual values
    # -- this lets us avoid having the labels overlap when the training set durations are close to each other
    # e.g., 30 and 45
    train_set_durs = sorted(curve_df['train_set_dur'].unique())
    dur_int_map = dict(zip(train_set_durs, range(len(train_set_durs))))
    curve_df['train_set_dur_ind'] = curve_df['train_set_dur'].map(dur_int_map)
    
    csv_path = results_root.joinpath(csv_filename)
    curve_df.to_csv(csv_path, index=False)


PROJ_ROOT = pyprojroot.here()
BR_RESULTS_ROOT = PROJ_ROOT / 'results' / 'BirdsongRecognition' / 'default_spect_params'


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--results_root',
                        help='root of directory containing reuslts for all birds in BirdsongRecognition',
                        default=BR_RESULTS_ROOT)
    parser.add_argument('--csv_filename',
                        help='filename of .csv that will be saved by this script in results_root',
                        default='error_across_birds_with_cleanup.csv')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(results_root=args.results_root, csv_filename=args.csv_filename)
