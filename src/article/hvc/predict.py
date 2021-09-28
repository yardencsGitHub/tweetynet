from pathlib import Path

import joblib

import numpy as np

import pandas as pd
from tqdm import tqdm


def predict(extract_csv_path,
            clf_path,
            predict_dst,
            labelset,
            split='test'):
    """generate predictions using a trained classifier,
    and save those predictions to a text file

    Parameters
    ----------
    extract_csv_path : str, pathlib.Path
        path to csv saved by ``article.hvc.extract``,
        with column ``features_path`` where elements are paths
        to files containing arrays of features.
    clf_path : str, pathlib.Path
        path to classifier saved by ``article.hvc.fit.save_clf``
    predict_dst : str, pathlib.Path
        where text file of predicted labels should be saved.
    labelset : set
        of labels.
        Used to map integer predictions to string labels.
    split : str
        split that should be used for predictions. Default is 'test'.

    Returns
    -------
    pred_path : pathlib.Path
        path to text file containing predicted labels.
    """
    import vak  # to avoid circular imports

    predict_dst = Path(predict_dst).expanduser().resolve()
    if not predict_dst.exists() or not predict_dst.is_dir():
        raise NotADirectoryError(
            f'predict_dst not found, or not recognized as a directory:\n{predict_dst}'
        )

    extract_df = pd.read_csv(extract_csv_path)
    extract_df = extract_df[extract_df.split == split]
    clf = joblib.load(clf_path)

    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    inverse_labelmap = {v: k for k, v in labelmap.items()}

    ftr_paths = extract_df.features_path.values.tolist()
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

    pred_path = predict_dst / (extract_csv_path.stem + f'.pred.txt')
    with pred_path.open('w') as fp:
        fp.writelines(y_pred_list)

    return pred_path
