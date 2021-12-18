from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm


def _labelset_to_inv_labelmap(labelset):
    import vak.labeled_timebins

    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)
    if any([len(label) > 1 for label in labelmap.keys()]):  # only re-map if necessary
        print('found labels with more than one character, mapping to single character to compute syllable error rate')
        # (to minimize chance of knock-on bugs)
        labelmap = vak.labeled_timebins._multi_char_labels_to_single_char(labelmap)
    inverse_labelmap = {v: k for k, v in labelmap.items()}
    return inverse_labelmap


def predict(extract_csv_path,
            clf_path,
            predict_dst,
            labelset,
            split='test'):
    """generate predictions using a trained classifier,
    given arrays of features extracted from syllables,
    and then map the predicted classes back to labels.

    Adds predicted label sequences in a column
    called 'pred_labels' to the ``pandas.DataFrame``
    loaded from ``extract_csv_path``,
    and then saves the updated ``DataFrame``
    as a csv file in ``predict_dst``.

    Parameters
    ----------
    extract_csv_path : str, pathlib.Path
        path to csv saved by ``article.hvc.extract``,
        with column ``features_path`` where elements are paths
        to files containing arrays of features.
    clf_path : str, pathlib.Path
        path to classifier saved by ``article.hvc.fit.save_clf``
    predict_dst : str, pathlib.Path
        destination where csv with added column
        of predicted labels should be saved.
    labelset : set
        of labels, used to map integer predictions
        back to string labels.
    split : str
        split that should be used for predictions. Default is 'test'.

    Returns
    -------
    pred_csv_path : pathlib.Path
        path to csv file with column 'pred_labels' that
        contains the predicted labels as strings
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

    inverse_labelmap = _labelset_to_inv_labelmap(labelset)

    pred_labels = []  # will add as column to df
    for ind in tqdm(extract_df.index):
        ftr_path = extract_df.features_path[ind]
        if ftr_path == 'None':  # because there were no segments after 'semi-automated cleaning'
            pred_labels.append('')
            continue

        ftr_df = pd.read_csv(ftr_path)
        # note we drop labels column and *then* get just values, i.e. array of features
        x_pred = ftr_df.drop(labels=['labels'], axis="columns").values
        y_pred = clf.predict(x_pred)
        pred_labels.append(
            # map array of class integers back to string of labels: [0, 1, 2] --> 'abc'
            ''.join([inverse_labelmap[el] for el in y_pred])
        )

    extract_df['pred_labels'] = pred_labels
    pred_csv_path = predict_dst / extract_csv_path.name
    extract_df.to_csv(pred_csv_path, index=False)

    return pred_csv_path
