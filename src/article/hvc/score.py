import numpy as np

import pandas as pd
import vak


def score(prep_csv_path,
          pred_paths):
    prep_df = pd.read_csv(prep_csv_path)
    annots = vak.annotation.from_df(prep_df)
    y_true = [''.join(annot.seq.labels.tolist()) for annot in annots]

    seg_error_rate = vak.metrics.SegmentErrorRate()

    scores = {}
    for preds_source, pred_path in pred_paths.items():
        with pred_path.open('r') as fp:
            y_pred = fp.read().splitlines()

        rates = np.array([
            seg_error_rate(y_pred, y_true)
            for y_pred, y_true in zip(y_pred, y_true)
        ])

        mean_segment_error_rate = np.array(rates).mean()

        scores[preds_source] = {
            'segment_error_rates': rates,
            'mean_segment_error_rate': rates.mean()
        }

    return scores
