from pathlib import Path

import joblib

import numpy as np

import pandas as pd
from tqdm import tqdm


def predict(prep_csv_path,
            resegment_csv_path,
            clf_path,
            predict_dst,
            labelset):
    """generate predictions
    using a trained classifier

    Parameters
    ----------
    prep_csv_path : str, pathlib.Path
        path to csv saved by ``vak prep``
    resegment_csv_path : str, pathlib.Path
        path to csv saved by ``article.hvc.resegment.resegment``
    clf_path : str, pathlib.Path
        path to classifier saved by ``article.hvc.fit.save_clf``
    predict_dst : str, pathlib.Path
        where text file of predicted labels should be saved.
    labelset : set
        of labels.
        Used to map integer predictions to string labels.

    Returns
    -------
    pred_paths : dict
        with keys {'ground_truth', 'resegment'}
        whose corresponding values are paths to
        text files containing predicted labels.
    """
    import vak  # to avoid circular imports

    predict_dst = Path(predict_dst).expanduser().resolve()
    if not predict_dst.exists() or not predict_dst.is_dir():
        raise NotADirectoryError(
            f'predict_dst not found, or not recognized as a directory:\n{predict_dst}'
        )

    prep_df = pd.read_csv(prep_csv_path)
    resegment_df = pd.read_csv(resegment_csv_path)
    clf = joblib.load(clf_path)

    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    inverse_labelmap = {v: k for k, v in labelmap.items()}

    pred_paths = {}
    # loop over ground truth and resegmented data, to make predictions for each
    for preds_source, df, csv_path in zip(
            ('ground_truth', 'resegment'),
            (prep_df, resegment_df),
            (prep_csv_path, resegment_csv_path),
    ):
        ftr_paths = df.features_path.values.tolist()
        ftr_dfs = []
        for row_num, ftr_path in enumerate(tqdm(ftr_paths)):
            ftr_df = pd.read_csv(ftr_path)
            # "foreign key" maps back to row of resegment_df
            # so we can figure out which predictions are for which row
            ftr_df['foreign_key'] = row_num
            ftr_dfs.append(ftr_df)
        ftr_df = pd.concat(ftr_dfs)

        x_pred = ftr_df.drop(labels=['labels', 'foreign_key'], axis="columns").values
        y_pred = clf.predict(x_pred)
        split_inds = np.nonzero(np.diff(ftr_df.foreign_key.values))[0]
        y_pred_list = np.split(y_pred, split_inds)
        y_pred_list = [
            ''.join([inverse_labelmap[el] for el in y_pred]) + "\n"
            for y_pred in y_pred_list
        ]

        pred_path = predict_dst / (csv_path.stem + f'.pred.txt')
        with pred_path.open('w') as fp:
            fp.writelines(y_pred_list)
        pred_paths[preds_source] = pred_path

    return pred_paths
