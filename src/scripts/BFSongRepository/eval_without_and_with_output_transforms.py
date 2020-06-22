from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import vak.device
import vak.files
import vak.labels as labelfuncs
from vak import config, io, models, transforms
from vak.datasets.vocal_dataset import VocalDataset
from vak.datasets.unannotated_dataset import UnannotatedDataset


def compute_metrics(metrics, y_true, y_pred, y_true_labels, y_pred_labels):
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


def metrics_df_from_toml_path(toml_path,
                              min_segment_dur,
                              device='cuda',
                              spect_key='s',
                              timebins_key='t'):
    """computes evaluation metrics on a dataset from a config.toml file

    computes the metrics without and with transforms used for prediction

    Parameters
    ----------
    toml_path
    min_segment_dur
    device
    spect_key
    timebins_key

    Returns
    -------
    df : pandas.Dataframe
    """
    toml_path = Path(toml_path)
    cfg = config.parse.from_toml(toml_path)

    model_config_map = config.models.map_from_path(toml_path, cfg.eval.models)

    spect_standardizer = joblib.load(cfg.eval.spect_scaler_path)

    # ---- make pred dataset to actually do predictions
    # will call model.predict() and get back dict with predictions and spect paths
    transform, target_transform = transforms.get_defaults('predict',
                                                          spect_standardizer,
                                                          window_size=cfg.dataloader.window_size,
                                                          return_padding_mask=False,
                                                          )

    pred_dataset = UnannotatedDataset.from_csv(csv_path=cfg.eval.csv_path,
                                               split='test',
                                               window_size=cfg.dataloader.window_size,
                                               spect_key=spect_key,
                                               timebins_key=timebins_key,
                                               transform=transform,
                                               )

    pred_data = torch.utils.data.DataLoader(dataset=pred_dataset,
                                            shuffle=False,
                                            batch_size=1,  # hard coding to make this work for now
                                            num_workers=cfg.eval.num_workers)

    # ---- make eval dataset that we'll use to compute metrics
    # each batch will give us dict with 'spect', 'annot' and 'spect_path'
    # we can use 'spect_path' to find prediction in pred_dict and then compare to target
    # dict also includes 'padding_mask' so we can "unpad" the prediction vectors
    item_transform = transforms.get_defaults('eval',
                                             spect_standardizer,
                                             window_size=cfg.dataloader.window_size,
                                             return_padding_mask=True,
                                             )

    with cfg.eval.labelmap_path.open('r') as f:
        labelmap = json.load(f)

    eval_dataset = VocalDataset.from_csv(csv_path=cfg.eval.csv_path,
                                         split='test',
                                         labelmap=labelmap,
                                         spect_key=spect_key,
                                         timebins_key=timebins_key,
                                         item_transform=item_transform,
                                         )

    eval_data = torch.utils.data.DataLoader(dataset=eval_dataset,
                                            shuffle=False,
                                            # batch size 1 because each spectrogram reshaped into a batch of windows
                                            batch_size=1,
                                            num_workers=cfg.eval.num_workers)

    # get timebin dur to use when converting labeled timebins to labels, onsets and offsets
    timebin_dur = io.dataframe.validate_and_get_timebin_dur(
        pd.read_csv(cfg.eval.csv_path)
    )

    input_shape = pred_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
    )

    if device is None:
        device = vak.device.get_default_device()

    records = defaultdict(list)  # will be used with pandas.DataFrame.from_records to make output csv
    to_long_tensor = transforms.ToLongTensor()

    for model_name, model in models_map.items():
        model.load(cfg.eval.checkpoint_path)
        metrics = model.metrics  # metric name -> callable map we use below in loop

        pred_dict = model.predict(pred_data=pred_data,
                                  device=device)

        progress_bar = tqdm(eval_data)
        for ind, batch in enumerate(progress_bar):
            y_true, padding_mask, spect_path = batch['annot'], batch['padding_mask'], batch['spect_path']
            # need to convert spect_path to tuple for match in call to index() below
            spect_path = tuple(spect_path)
            records['spect_path'].append(spect_path[0])  # remove str from tuple
            y_true = y_true.to(device)
            y_true_np = np.squeeze(y_true.cpu().numpy())
            y_true_labels, _, _ = labelfuncs.lbl_tb2segments(y_true_np,
                                                             labelmap=labelmap,
                                                             timebin_dur=timebin_dur)
            y_true_labels = ''.join(y_true_labels.tolist())

            y_pred_ind = pred_dict['y'].index(spect_path)
            y_pred = pred_dict['y_pred'][y_pred_ind]
            y_pred = torch.argmax(y_pred, dim=1)  # assumes class dimension is 1
            y_pred = torch.flatten(y_pred)
            y_pred = y_pred.unsqueeze(0)[padding_mask]
            y_pred_np = np.squeeze(y_pred.cpu().numpy())
            y_pred_labels, _, _ = labelfuncs.lbl_tb2segments(y_pred_np,
                                                             labelmap=labelmap,
                                                             timebin_dur=timebin_dur,
                                                             min_segment_dur=None,
                                                             majority_vote=False)
            y_pred_labels = ''.join(y_pred_labels.tolist())

            metric_vals_batch = compute_metrics(metrics, y_true, y_pred, y_true_labels, y_pred_labels)
            for metric_name, metric_val in metric_vals_batch.items():
                records[metric_name].append(metric_val)

            # --- apply majority vote and min segment dur transforms separately
            # need segment_inds_list for both transforms
            segment_inds_list = labelfuncs.lbl_tb_segment_inds_list(y_pred_np,
                                                                    unlabeled_label=labelmap['unlabeled'])

            # ---- majority vote transform
            y_pred_np_mv = labelfuncs.majority_vote_transform(y_pred_np, segment_inds_list)
            y_pred_mv = to_long_tensor(y_pred_np_mv).to(device)
            y_pred_mv_labels, _, _ = labelfuncs.lbl_tb2segments(y_pred_np_mv,
                                                                labelmap=labelmap,
                                                                timebin_dur=timebin_dur,
                                                                min_segment_dur=None,
                                                                majority_vote=False)
            y_pred_mv_labels = ''.join(y_pred_mv_labels.tolist())
            metric_vals_batch_mv = compute_metrics(metrics, y_true, y_pred_mv,
                                                   y_true_labels, y_pred_mv_labels)
            for metric_name, metric_val in metric_vals_batch_mv.items():
                records[f'{metric_name}_majority_vote'].append(metric_val)

            # ---- min segment dur transform
            y_pred_np_mindur, _ = labelfuncs.remove_short_segments(y_pred_np,
                                                                   segment_inds_list,
                                                                   timebin_dur=timebin_dur,
                                                                   min_segment_dur=min_segment_dur,
                                                                   unlabeled_label=labelmap['unlabeled'])
            y_pred_mindur = to_long_tensor(y_pred_np_mindur).to(device)
            y_pred_mindur_labels, _, _ = labelfuncs.lbl_tb2segments(y_pred_np_mindur,
                                                                    labelmap=labelmap,
                                                                    timebin_dur=timebin_dur,
                                                                    min_segment_dur=None,
                                                                    majority_vote=False)
            y_pred_mindur_labels = ''.join(y_pred_mindur_labels.tolist())
            metric_vals_batch_mindur = compute_metrics(metrics, y_true, y_pred_mindur,
                                                       y_true_labels, y_pred_mindur_labels)
            for metric_name, metric_val in metric_vals_batch_mindur.items():
                records[f'{metric_name}_min_segment_dur'].append(metric_val)

            # ---- and finally both transforms, in same order we apply for prediction
            y_pred_np_mindur_mv, segment_inds_list = labelfuncs.remove_short_segments(y_pred_np,
                                                                                      segment_inds_list,
                                                                                      timebin_dur=timebin_dur,
                                                                                      min_segment_dur=min_segment_dur,
                                                                                      unlabeled_label=labelmap[
                                                                                          'unlabeled'])
            y_pred_np_mindur_mv = labelfuncs.majority_vote_transform(y_pred_np_mindur_mv,
                                                                     segment_inds_list)
            y_pred_mindur_mv = to_long_tensor(y_pred_np_mindur_mv).to(device)
            y_pred_mindur_mv_labels, _, _ = labelfuncs.lbl_tb2segments(y_pred_np_mindur_mv,
                                                                       labelmap=labelmap,
                                                                       timebin_dur=timebin_dur,
                                                                       min_segment_dur=None,
                                                                       majority_vote=False)
            y_pred_mindur_mv_labels = ''.join(y_pred_mindur_mv_labels.tolist())
            metric_vals_batch_mindur_mv = compute_metrics(metrics, y_true, y_pred_mindur_mv,
                                                          y_true_labels, y_pred_mindur_mv_labels)
            for metric_name, metric_val in metric_vals_batch_mindur_mv.items():
                records[f'{metric_name}_min_dur_maj_vote'].append(metric_val)

        df = pd.DataFrame.from_records(records)
        return df


CONFIG_ROOT = Path('src/configs/BFSongRepository')
BIRD_ID_MIN_SEGMENT_DUR_MAP = {
    'bl26lb16': 0.02,
    'gr41rd51': 0.02,
    'gy6or6': 0.01,
    'or60yw70': 0.02,
}


def main():
    for bird_id, min_segment_dur in BIRD_ID_MIN_SEGMENT_DUR_MAP.items():
        toml_root = CONFIG_ROOT.joinpath(bird_id)
        eval_toml_paths = sorted(toml_root.glob('**/*eval*toml'))

        all_dfs = []
        for eval_toml_path in eval_toml_paths:
            print(f'computing metrics from dataset in .toml file: {eval_toml_path.name}')
            toml_df = metrics_df_from_toml_path(eval_toml_path, min_segment_dur)
            date = eval_toml_path.parents[0].name  # directory name is date
            print(f'date for this .toml file: {date}')
            toml_df['date'] = date
            all_dfs.append(toml_df)

        output_df = pd.concat(all_dfs)
        output_df['bird_id'] = bird_id
        print(f"adding 'bird_id' to concatenated data frames: {bird_id}")
        csv_fname = f'{bird_id}.metrics.csv'
        csv_path = Path('results/BFSongRepository').joinpath(csv_fname)
        print(f'saving csv as: {csv_path}')
        output_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
