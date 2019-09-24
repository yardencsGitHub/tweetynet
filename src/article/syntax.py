"""functions used for syntax analysis"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent


def date_from_cbin_filename(cbin_filename):
    """extracts date as a string from a .cbin filename
    and converts to a datetime object

    Parameters
    ----------
    cbin_filename : str
        path to a .cbin file

    Returns
    -------
    dt : datetime
        convert from string using strptime function

    Examples
    --------
    >>> cbin_filename = 'gy6or6_baseline_220312_0836.3.cbin'
    >>> dt = date_from_cbin_filename(cbin_filename)
    """
    name = Path(cbin_filename).name
    splitname = name.split('.')  # splits into: stem, file number, extension
    splitname = splitname[0]  # just keep the stem
    splitname = splitname.split('_')  # further split stem; last two will be date + time
    date, time = splitname[-2:]
    dt = date + '-' + time
    dt = datetime.strptime(dt, '%d%m%y-%H%M')
    return dt


FIELDS_SYNTAX = [
    'date',
    'time',
    'cbin_filename',
    'label',
    'label_plus_one',
    'label_minus_one',
    'label_minus_two',
]


def make_df_trans_probs(vds_list, start_label='S', end_label='E'):
    """make a dataframe where each row represents a transition
    from one label to another, e.g. in annotated birdsong.

    Can be used to compute transition probabilities between birdsong syllables.

    Takes a list of vak Datasets; as written now, the function expects each
    vocalization in the dataset to be from .cbin audio files.


    Parameters
    ----------
    vds_list : list
        of vak.Dataset.
    start_label : str
        Label that represents start of a sequence, e.g. a song bout. Default is 'S'.
        The function adds a transition from this start state to the first labeled
        segment for every sequence, i.e. every vocalization in a dataset.
    end_label : str
        Label that represents end of a sequence. Default is 'E'.
        The function adds a transition from the first labeled segemtn to this end state
        for every sequence, i.e. every vocalization in a dataset.

    Returns
    -------
    df : pandas.Dataframe
        where each row represents one transition from one label to another
        and has the following columns:
            date : datetime
                extracted from audio filename.
                Used to determine transition probabilities on a certain day,
                and for resampling analysis where the dates are shuffled.
            cbin_filename : str
                filename of audio file associated with annotation.
            label : str
                String label for current label of interest.
                For example, label 'a' in transition 'a' -> 'b'.
            label_plus_one : str
                Label that follows current label.
                For exapmle, label 'b' in transition 'a' -> 'b'.
            label_minus_one : str
                Label that preceded current label.
            label_minus_two : str
                Label that preceded label before current label.
    """
    df_dict = {field: [] for field in FIELDS_SYNTAX}
    for vds in vds_list:
        for voc in vds.voc_list:
            cbin_filename = voc.audio_path
            datetime = date_from_cbin_filename(cbin_filename)
            labels = voc.annot.labels.tolist()
            labels = [start_label] + labels + [end_label]
            len_labels = len(labels)
            for ind, label in enumerate(labels):
                if ind == 0:
                    label_minus_one = None
                    label_minus_two = None
                    label_plus_one = labels[ind + 1]
                elif ind == 1:
                    label_minus_one = labels[0]
                    label_minus_two = None
                    label_plus_one = labels[ind + 1]
                elif ind == (len_labels - 1):  # i.e., last index
                    label_minus_one = labels[ind - 1]
                    label_minus_two = labels[ind - 2]
                    label_plus_one = None
                else:
                    label_minus_one = labels[ind - 1]
                    label_minus_two = labels[ind - 2]
                    label_plus_one = labels[ind + 1]
                df_dict['date'].append(datetime.date())
                df_dict['time'].append(datetime.time())
                df_dict['cbin_filename'].append(cbin_filename)
                df_dict['label'].append(label)
                df_dict['label_plus_one'].append(label_plus_one)
                df_dict['label_minus_one'].append(label_minus_one)
                df_dict['label_minus_two'].append(label_minus_two)
    df = pd.DataFrame.from_dict(df_dict)
    return df


def get_trans_prob(df, date, label, label_plus_one):
    """get probability of transition from one label to another,
    given a dataframe where rows are transitions

    Parameters
    ----------
    df : pandas.Dataframe
        returned by the make_df_trans_probs function
    date : datetime.date
        date to use to create the transition matrix.
        Must occur in the 'date' column of the dataframe
    label : str
        we want to compute the probability of
        transitioning from this label to label_plus_one
    label_plus_one : str
        label we want to compute the probability of
        transitioning to, from label_plus_one

    Returns
    -------
    p : float
        probability of transitioning from label to label_plus_one on
        the specified date, computed using the dataframe
    """
    df_date = df[df['date'] == date]
    label_count = len(
        df_date[df_date['label'] == label].index
    )
    trans_count = len(
        df_date[(df_date['label'] == label) & (df_date['label_plus_one'] == label_plus_one)].index
    )
    if label_count > 0:
        p = trans_count / label_count
    else:
        p = 0.
    return p


def make_trans_mat(df, date, min_p=0.01):
    """make transition matrix from a dataframe
    (returned by the make_df_trans_probs function)

    Parameters
    ----------
    df : pandas.Dataframe
        returned by the make_df_trans_probs function
    date : datetime.date
        date to use to create the transition matrix.
        Must occur in the 'date' column of the dataframe
    min_p : float
        minimum transition probability to keep.
        Default is 0.01. Any values below that will be
        set to 0.0, and the rows will be normalized so
        that they sum to 1.0.

    Returns
    -------
    trans_mat : numpy.ndarray
        transition matrix

    Notes
    -----
    Each row corresponds to one label from the unique set found
    in the dataframe, and the values in that row represent the
    probability of transitioning from that label to the label
    that corresponds to the column in which the value is found.
    """
    labels = df['label'].unique()
    num_labels = labels.shape[0]
    trans_mat = np.zeros((num_labels, num_labels))
    for row, label in enumerate(labels):
        if label == 'E':
            continue
        else:
            for col, label_plus_one in enumerate(labels):
                p = get_trans_prob(df, date, label, label_plus_one)
                if p > min_p:
                    trans_mat[row, col] = p
                else:
                    trans_mat[row, col] = 0.
    # adjust so all rows sum to 1
    row_sums = trans_mat.sum(axis=1)
    trans_mat = trans_mat[row_sums != 0.0, :]
    row_sums = trans_mat.sum(axis=1)
    trans_mat = trans_mat / row_sums[:, np.newaxis]
    return trans_mat


def find_branch_points(trans_mat, labels):
    """given a 2-d transition matrix, find branch points, i.e. labels
    that can transition to more than one label

    Parameters
    ----------
    trans_mat : numpy.ndarray
        2-d matrix where rows represent probabilities of transitioning
        from some label to whichever label is indexed by the column
        where the probability appears.
    labels : list
        of str, unique set of labels. The index of each label
        must correspond to the row in trans_mat representing the
        probabilities of transitioning from that label to others.

    Returns
    -------
    branch_point_ind : numpy.ndarray
        indices of rows that are branch points in trans_mat.
    branch_point_lbl : list
        of labels. Labels for indices in branch_point_ind.
    """
    branch_point_ind = []
    branch_point_lbl = []
    for ind, (row, label) in enumerate(zip(trans_mat, labels)):
        ps = np.nonzero(row)[0]
        if ps.shape[0] > 1:
            branch_point_ind.append(ind)
            branch_point_lbl.append(label)
    if branch_point_ind:  # is not empty
        branch_point_ind = np.asarray(branch_point_ind)
    return branch_point_ind, branch_point_lbl


