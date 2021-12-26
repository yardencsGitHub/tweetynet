def pred2labels(y_pred_np,
                labelmap,
                t,
                timebin_dur,
                cleanup_type='none',
                min_segment_dur=None):
    """convert neural network predictions
    from a vector of labeled time bins
    to a sequence of strings

    Parameters
    ----------
    y_pred_np : numpy.ndarray
        vector of labeled timebins,
        output of a neural network
    labelmap : dictionary
        that maps string labels to
        integer classes that are
        output by network
    t : numpy.ndarray
        vector of same size as ``y_pred_np``,
        values are times at center of bins
        for time bins in a spectrogram.
    timebin_dur : float
        duration of timebin in seconds
    cleanup_type : str
        type of clean-up to apply.
        One of {'none', 'majority_vote',
        'min_segment_dur', 'min_segment_dur_majority_vote'}.
    min_segment_dur : float
        duration of minimum segment.
        If specified

    Returns
    -------
    y_pred_np : numpy.ndarray
        with clean-up applied, if any
    y_pred_labels : list
        predicted segment labels
    pred_onsets_s : numpy.ndarray
        predict segment onset times, in seconds
    pred_offsets_s : numpy.ndarray
        predict segment offset times, in seconds
    """
    # import vak in function to avoid circular imports
    from vak import transforms
    from vak.labeled_timebins import (
        lbl_tb2segments,
        majority_vote_transform,
        lbl_tb_segment_inds_list,
        remove_short_segments
    )

    if cleanup_type == 'none':
        y_pred_labels, pred_onsets_s, pred_offsets_s = lbl_tb2segments(y_pred_np,
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
            y_pred_labels, pred_onsets_s, pred_offsets_s = lbl_tb2segments(y_pred_np,
                                                                           labelmap=labelmap,
                                                                           t=t,
                                                                           min_segment_dur=None,
                                                                           majority_vote=False)
        elif cleanup_type == 'min_segment_dur':
            y_pred_np, _ = remove_short_segments(y_pred_np,
                                                 segment_inds_list,
                                                 timebin_dur=timebin_dur,
                                                 min_segment_dur=min_segment_dur,
                                                 unlabeled_label=labelmap['unlabeled'])
            y_pred_labels, pred_onsets_s, pred_offsets_s = lbl_tb2segments(y_pred_np,
                                                                           labelmap=labelmap,
                                                                           t=t,
                                                                           min_segment_dur=None,
                                                                           majority_vote=False)
        elif cleanup_type == 'min_segment_dur_majority_vote':
            y_pred_np, segment_inds_list = remove_short_segments(y_pred_np,
                                                                 segment_inds_list,
                                                                 timebin_dur=timebin_dur,
                                                                 min_segment_dur=min_segment_dur,
                                                                 unlabeled_label=labelmap['unlabeled'])
            y_pred_np = majority_vote_transform(y_pred_np,
                                                segment_inds_list)

            y_pred_labels, pred_onsets_s, pred_offsets_s = lbl_tb2segments(y_pred_np,
                                                                           labelmap=labelmap,
                                                                           t=t,
                                                                           min_segment_dur=None,
                                                                           majority_vote=False)
        # do this at end regardless of cleanup type -- to obey DRY
        y_pred_labels = ''.join(y_pred_labels.tolist())

    return y_pred_np, y_pred_labels, pred_onsets_s, pred_offsets_s
