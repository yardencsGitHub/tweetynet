from collections import namedtuple

import numpy as np
import pandas as pd


SegmentErrorRates = namedtuple(typename='SegmentErrorRates',
                               field_names=('segment_error_rates', 'mean_segment_error_rate'))


def segment_error_rate(pred_csv_path,
                       ground_truth_csv_path=None,
                       split='test'):
    """compute segment error rates--
    i.e. a normalized edit distance--
    given ground truth labels and
    the labels predicted using
    features extracted from segments.

    Parameters
    ----------
    pred_csv_path : str, pathlib.Path
        path to csv returned by ``article.hvc.predict``.
        This csv will have the column 'pred_labels'
        that in each row contains a string of predicted labels
        for the features from a file pointed to
        by the 'features_path' column in the same row.
        It may also have the ground truth annotation
        in the file pointed to by the 'annot_path' column of that row.
    ground_truth_csv_path : str, pathlib.Path
        path to csv with ground truth labels,
        in file(s) pointed to by the 'annot_path' column.
        Default is None. If None, then the annotations in
        column with the same name from the ``pred_csv_path``
        are used. This argument is used to provide another .csv
        when the annotations are not ground truth in the ``pred_csv_path``,
        e.g., because the audio was re-segmented by
        ``article.hvc.resegment``.
    split : str
        split that should be used to compute error rates. Default is 'test'.
        After loading the csv into a ``pandas.DataFrame``,
        the only rows of the ``DataFrame`` that are kept are those
        where the 'split' column equals the specified split.

    Returns
    -------
    seg_error_tuple : named tuple
        with fields 'segment_error_rates' and 'mean_segment_error_rate'.
        The 'segment_error_rates' is a list of values,
        one for each line in the .txt annotation files.
        The 'mean_segment_error_rate' is the mean across
        the 'segment_error_rates' list.
    """
    import vak

    pred_df = pd.read_csv(pred_csv_path)
    if ground_truth_csv_path is not None:
        gt_df = pd.read_csv(ground_truth_csv_path)
    else:
        gt_df = None

    pred_df = pred_df[pred_df.split == split]
    if gt_df is not None:
        gt_df = gt_df[gt_df.split == split]
        if not all(
            [pred_audio == gt_audio
             for pred_audio, gt_audio in zip(pred_df.audio_path.values, gt_df.audio_path.values)]
        ):
            raise ValueError(
                'not all audio paths matched in predicted and ground truth csvs '
                '-- check specified paths.\n'
                f'pred_csv_path: {pred_csv_path}\n'
                f'ground_truth_csv_path: {ground_truth_csv_path}'
            )

    if gt_df is not None:
        annots = vak.annotation.from_df(gt_df)
        y_true = [''.join(annot.seq.labels.tolist()) for annot in annots]
    else:
        annots = vak.annotation.from_df(pred_df)
        y_true = [''.join(annot.seq.labels.tolist()) for annot in annots]

    y_pred = pred_df.pred_labels.values.tolist()
    y_pred = [yp
              if isinstance(yp, str) else ''  # convert np.nan back to empty string
              for yp in y_pred]

    if not len(y_true) == len(y_pred):
        raise ValueError(
            f'number of sequences in `y_true`, {len(y_true)}, '
            f'did not equal number in `y_pred`, {len(y_pred)}'
        )

    seg_error_rate = vak.metrics.SegmentErrorRate()

    rates = np.array([
        seg_error_rate(y_pred, y_true)
        for y_pred, y_true in zip(y_pred, y_true)
    ])

    seg_error_tuple = SegmentErrorRates(segment_error_rates=rates,
                                        mean_segment_error_rate=rates.mean())

    return seg_error_tuple
