from collections import defaultdict

import numpy as np
import torch


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
