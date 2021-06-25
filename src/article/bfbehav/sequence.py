"""functions for analyzing sequences of Bengalese finch syllable labels"""
from collections import Counter, namedtuple

import numpy as np


def states_and_counts_from_annots(annots):
    """get ``states`` and ``counts``
    from a list of ``crowsetta.Annotation``s,
    to build a transition matrix
    showing the probability of going from
    one state to another.

    Parameters
    ----------
    annots : list
        of ``crowsetta.Annotation``s
        whose ``Sequence.label`` attributes
        will be used to count transitions
        from one label to another
        in the sequences of syllables

    Returns
    -------
    states : set
        of characters, set of
        unique labels
        that occur in the list of
        ``Annotation``s.
    counts : collections.Counter
        where keys are transitions
        of the form "from label,
        to label", e.g. "ab",
        and values are number of
        occurrences of that transition.
    """
    states = sorted(
        set([lbl
             for annot in annots
             for lbl in annot.seq.labels.tolist()])
    )

    counts = Counter()
    for annot in annots:
        labels = annot.seq.labels.tolist()
        trans = zip(labels[:-1], labels[1:])
        for a_trans in trans:
            counts[a_trans] += 1
    return states, counts


def row_norm(mat):
    return mat / mat.sum(axis=1)[:, np.newaxis]


TransitionMatrix = namedtuple('TransitionMatrix',
                              field_names=('counts',
                                           'matrix',
                                           'states'))


def transmat_from_annots(annots,
                         thresh=None):
    """from list of ``crowsetta.Annotation``s

    returns ``TransitionMatrix`` tuple with fields
    'counts', 'matrix', and 'states'


    Parameters
    ----------
    annots : list
        of ``crowsetta.Annotation``s
    thresh : float
        threshold used to smooth probabilities.
        If not None, any probabilities less
        than threshold are set to 0.0 and then
        the transition matrix is again row
        normalized

    Returns
    -------
    trans_mat : TransitionMatrix
    """
    states, counts = states_and_counts_from_annots(annots)

    counts_arr = np.array(
        [[counts[i, j] for j in states] for i in states], dtype=float
    )

    trans_mat = row_norm(counts_arr)
    if thresh is not None:
        trans_mat[trans_mat < thresh] = 0.0
        trans_mat = row_norm(trans_mat)

    return TransitionMatrix(counts=counts, matrix=trans_mat, states=states)
