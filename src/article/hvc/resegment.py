from collections import defaultdict
from pathlib import Path

import crowsetta
import evfuncs
import numpy as np
import pandas as pd
import vak.annotation
from tqdm import tqdm


def resegment(prep_csv,
              segment_params,
              annot_dst,
              csv_dst,
              dummy_label='-',
              split=None):
    """re-segment a split from a dataset,
    and save these segments as annotations
    with dummy labels.

    This function produces two types of segmentation:
        - "raw", which is what you would get
        if blindly segmenting unlabeled data
        without carefully cleaning up by hand.
        - "some cleaning", which simulates
        semi-automated clean-up, by keeping only
        the raw segments that occur between the
        first onset and the last offset in the
        ground truth, manually-cleaned data.

    Parameters
    ----------
    prep_csv : str, pathlib.Path
        path to a .csv prepared by ``vak``,
        e.g. during a run of ``learncurve``,
        that represents a dataset
    segment_params : dict
        of parameters to use.
        Keys and values as expected by
        the function ``evfuncs.segment_song``
    annot_dst : str, pathlib.Path
        destination for annotations.
        Annotations are saved as .csv files
        in the generic ``crowsetta`` format.
    csv_dst : str, pathlib.Path
        destination for .csv files
        that represent the dataset. Same
        dataset as ``prep_csv``, but with
        the annotations created
        by re-segmenting.
    dummy_label : str
        single character, dummy label given
        to all segments after re-segmenting.
        Default is '-'.
    split : str
        split that should be re-segmented.
        If specified, the .csv files this
        function generate will
        only contain this split.

    Returns
    -------
    resegment_csv_paths : dict
        with keys {'not-cleaned', 'semi-automated-cleaning'}
        that map to paths to csv files
        which were saved in ``csv_dst``

    Notes
    -----
    Uses the function ``evfuncs.segment_song``
    which replicates the behavior of the ``evsonganaly``
    GUI used to annotate the
    Bengalese Finch Song Repository.
    """
    SEGMENTATION_TYPES = ('not-cleaned', 'semi-automated-cleaning')

    prep_csv = Path(prep_csv)
    annot_dst = Path(annot_dst)
    csv_dst = Path(csv_dst)

    # ---- make paths to save new annotations here, fail early if for some reason it doesn't work
    annot_csv_paths = {}
    for segmentation in SEGMENTATION_TYPES:
        annot_path = annot_dst / (prep_csv.stem + f'.{segmentation}.annot.csv')
        annot_path = annot_path.resolve()
        annot_csv_paths[segmentation] = annot_path

    prep_df = pd.read_csv(prep_csv)
    if split is not None:
        prep_df = prep_df[prep_df.split == split]

    prep_annots = vak.annotation.from_df(prep_df)

    audio_paths = prep_df.audio_path.values
    annots_by_segmentation = defaultdict(list)  # of generated files, will add to csvs below

    pbar = tqdm(zip(audio_paths, prep_annots))
    n_audio = len(audio_paths)
    for ind, (audio_path, ground_truth_annot) in enumerate(pbar):
        if not Path(audio_path).name == Path(ground_truth_annot.audio_path).name:
            raise ValueError(
                f'`audio_path` number {ind} did not match path from annotation.\n'
                f'audio_path: {audio_path}'
                f'annotation audio path: {ground_truth_annot.audio_path}'
            )
        pbar.set_description(
            f'resegmenting audio file {ind + 1} of {n_audio}:{Path(audio_path).name}'
        )
        rawsong, samp_freq = evfuncs.load_cbin(audio_path)
        smooth = evfuncs.smooth_data(rawsong, samp_freq)
        onsets_s, offsets_s = evfuncs.segment_song(smooth, samp_freq, **segment_params)
        labels = np.array(list(dummy_label * onsets_s.shape[0]))  # dummy labels

        raw_seq = crowsetta.Sequence.from_keyword(onsets_s=onsets_s, offsets_s=offsets_s, labels=labels)
        raw_annot = crowsetta.Annotation(annot_path=annot_csv_paths['not-cleaned'],
                                         audio_path=audio_path,
                                         seq=raw_seq)
        annots_by_segmentation['not-cleaned'].append(raw_annot)

        # simulate semi-automated cleaning of segments, using ground truth data:
        # keep all the onsets and offsets for which it is true that the onset is
        # is larger than the first onset in the cleaned ground truth data **and**
        # the offset is less than the last offset in the cleaned groudn truth data
        are_between_gt_onset_and_offset = np.logical_and(
            onsets_s > ground_truth_annot.seq.onsets_s[0],
            offsets_s < ground_truth_annot.seq.offsets_s[-1],
        )
        onsets_s = onsets_s[are_between_gt_onset_and_offset]
        offsets_s = offsets_s[are_between_gt_onset_and_offset]
        labels = np.array(list(dummy_label * onsets_s.shape[0]))  # re-make dummy labels

        someclean_seq = crowsetta.Sequence.from_keyword(onsets_s=onsets_s, offsets_s=offsets_s, labels=labels)
        someclean_annot = crowsetta.Annotation(annot_path=annot_csv_paths['semi-automated-cleaning'],
                                               audio_path=audio_path,
                                               seq=someclean_seq)
        annots_by_segmentation['semi-automated-cleaning'].append(someclean_annot)

    # ---- save annotation
    scribe = crowsetta.Transcriber(format='csv')

    resegment_csv_paths = {}
    for segmentation in SEGMENTATION_TYPES:
        annots = annots_by_segmentation[segmentation]
        annot_csv_path = str(annot_csv_paths[segmentation])
        scribe.to_csv(annots, annot_csv_path)

        annot_paths = [annot_csv_path for _ in audio_paths]  # all the same_annot path, same len as audio_paths
        df = prep_df.copy()
        df['annot_path'] = annot_paths
        df['annot_format'] = 'csv'
        reseg_csv_path = csv_dst / (prep_csv.stem + f'.{segmentation}.csv')
        df.to_csv(reseg_csv_path, index=False)
        resegment_csv_paths[segmentation] = reseg_csv_path

    return resegment_csv_paths
