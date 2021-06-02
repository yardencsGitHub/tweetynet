from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vak import config, io, models, transforms
from vak.datasets.vocal_dataset import VocalDataset
import vak.device
import vak.files
from vak.labeled_timebins import (
    lbl_tb2segments,
    majority_vote_transform,
    lbl_tb_segment_inds_list,
    remove_short_segments
)
from vak.core.learncurve import train_dur_csv_paths as _train_dur_csv_paths
from vak.logging import log_or_print

from .metrics import compute_metrics, boundary_err


# number of timebins from an onset or offset
# in which we count errors involving "unlabeled" / silent gaps
N_TIMEBINS_FROM_ONOFFSET = 2

CLEANUP_TYPES = ('none',
                 'majority_vote',
                 'min_segment_dur',
                 'min_segment_dur_majority_vote')


def eval_with_output_tfms(csv_path,
                          model_config_map,
                          checkpoint_path,
                          labelmap,
                          window_size,
                          num_workers,
                          min_segment_dur,
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
            for cleanup_type in CLEANUP_TYPES:
                # append this first at beginning of code block
                records['cleanup'].append(cleanup_type)

                y_true, padding_mask, spect_path = batch['annot'], batch['padding_mask'], batch['spect_path']
                if isinstance(spect_path, list) and len(spect_path) == 1:
                    spect_path = spect_path[0]  # __getitem__ returns 1-element list for paths (with batch size 1)
                t = vak.files.spect.load(spect_path)[timebins_key]
                if len(t)==1: # this is added b.c. when loading canary npz files the data is in [[]]
                    t=t[0]
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
                else:
                    # need segment_inds_list for transforms
                    segment_inds_list = lbl_tb_segment_inds_list(y_pred_np,
                                                                 unlabeled_label=labelmap['unlabeled'])

                    if cleanup_type == 'majority_vote':
                        y_pred_np = majority_vote_transform(y_pred_np, segment_inds_list)
                        y_pred = to_long_tensor(y_pred_np).to(device)
                        y_pred_labels, _, _ = lbl_tb2segments(y_pred_np,
                                                              labelmap=labelmap,
                                                              t=t,
                                                              min_segment_dur=None,
                                                              majority_vote=False)
                        y_pred_labels = ''.join(y_pred_labels.tolist())

                    elif cleanup_type == 'min_segment_dur':
                        y_pred_np, _ = remove_short_segments(y_pred_np,
                                                             segment_inds_list,
                                                             timebin_dur=timebin_dur,
                                                             min_segment_dur=min_segment_dur,
                                                             unlabeled_label=labelmap['unlabeled'])
                        y_pred = to_long_tensor(y_pred_np).to(device)
                        y_pred_labels, _, _ = lbl_tb2segments(y_pred_np,
                                                              labelmap=labelmap,
                                                              t=t,
                                                              min_segment_dur=None,
                                                              majority_vote=False)
                        y_pred_labels = ''.join(y_pred_labels.tolist())

                    elif cleanup_type == 'min_segment_dur_majority_vote':
                        y_pred_np, segment_inds_list = remove_short_segments(y_pred_np,
                                                                             segment_inds_list,
                                                                             timebin_dur=timebin_dur,
                                                                             min_segment_dur=min_segment_dur,
                                                                             unlabeled_label=labelmap['unlabeled'])
                        y_pred_np = majority_vote_transform(y_pred_np,
                                                            segment_inds_list)
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


def learncurve_with_transforms(previous_run_path,
                               min_segment_dur,
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
                                                spect_scaler_path=spect_scaler_path,
                                                min_segment_dur=min_segment_dur)
                eval_df['train_set_dur'] = train_dur
                eval_df['replicate_num'] = replicate_num
                eval_df['model_name'] = model_name
                eval_dfs.append(eval_df)

    eval_dfs = pd.concat(eval_dfs)
    return eval_dfs


def train_with_transforms(results_root,
                          min_segment_dur,
                          logger=None):
    results_root = Path(results_root)
    toml_path = sorted(results_root.glob('*.toml'))
    assert len(toml_path) == 1, f'found more than one .toml config file: {toml_path}'
    toml_path = toml_path[0]

    cfg = config.parse.from_toml_path(toml_path)

    # ---- get all the parameters from the config we need
    model_config_map = config.models.map_from_path(toml_path, cfg.eval.models)
    csv_path = cfg.eval.csv_path
    window_size = cfg.dataloader.window_size
    num_workers = cfg.eval.num_workers
    normalize_spectrograms = cfg.train.normalize_spectrograms

    log_or_print(
        f"using dataset from .csv file: {csv_path}",
        logger=logger,
        level="info",
    )

    if normalize_spectrograms:
        spect_scaler_path = cfg.eval.spect_scaler_path
        log_or_print(
            f"Using spect scaler to normalize: {spect_scaler_path}",
            logger=logger,
            level="info",
        )
    else:
        spect_scaler_path = None

    # ---- have to load labelmap before we can get models
    labelmap_path = cfg.eval.labelmap_path
    log_or_print(
        f"Using labelmap: {labelmap_path}", logger=logger, level="info"
    )
    with labelmap_path.open("r") as f:
        labelmap = json.load(f)

    eval_dfs = []  # hack to make loop over items work -- we only ever eval one model
    for model_name, model_config in model_config_map.items():
        log_or_print(
            f"Evaluating model: {model_name}", logger=logger, level="info"
        )
        ckpt_path = cfg.eval.checkpoint_path
        log_or_print(
            f"Using checkpoint: {ckpt_path}", logger=logger, level="info"
        )

        eval_df = eval_with_output_tfms(csv_path,
                                        model_config_map,
                                        checkpoint_path=ckpt_path,
                                        labelmap=labelmap,
                                        window_size=window_size,
                                        num_workers=num_workers,
                                        spect_scaler_path=spect_scaler_path,
                                        min_segment_dur=min_segment_dur)
        eval_df['model_name'] = model_name
        eval_dfs.append(eval_df)

    eval_dfs = pd.concat(eval_dfs)
    return eval_dfs
