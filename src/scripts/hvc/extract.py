import argparse
from pathlib import Path

import dask
import hvc
from hvc.audiofileIO import Syllable
from hvc.features.feature_dicts import (
    single_syl_features_switch_case_dict,
    multiple_syl_features_switch_case_dict
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import vak
import vak.converters


def make_syls(
    raw_audio,
    samp_freq,
    spect_maker,
    labels,
    onsets_Hz,
    offsets_Hz,
):
    """Make spectrograms from syllables.
    This method isolates making spectrograms from selecting syllables
    to use so that spectrograms can be loaded 'lazily', e.g., if only
    duration features are being extracted that don't require spectrograms.

    Parameters
    ----------
    raw_audio : ndarray
    samp_freq : int
    labels : str, list, or ndarray
    onsetz_Hz : ndarray
    offsets_Hz : ndarray
    """
    if type(labels) not in [str, list, np.ndarray]:
        raise TypeError(
            "labels must be of type str, list, or numpy ndarray, " "not {}".type(labels)
        )

    if type(labels) is str:
        labels = list(labels)

    if type(labels) is list:
        labels = np.asarray(labels)

    @dask.delayed
    def _make_syl(label, onset, offset, ind):
        syl_audio = raw_audio[onset:offset]

        try:
            spect, freq_bins, time_bins = spect_maker.make(syl_audio, samp_freq)
        except WindowError as err:
            spect, freq_bins, time_bins = (np.nan, np.nan, np.nan)

        syl = Syllable(
            syl_audio=syl_audio,
            samp_freq=samp_freq,
            spect=spect,
            nfft=spect_maker.nperseg,
            overlap=spect_maker.noverlap,
            freq_cutoffs=spect_maker.freqCutoffs,
            freq_bins=freq_bins,
            time_bins=time_bins,
            index=ind,
            label=label,
        )
        return syl

    syls = []
    for ind, (label, onset, offset) in enumerate(zip(labels, onsets_Hz, offsets_Hz)):
        syl = _make_syl(label, onset, offset, ind)
        syls.append(syl)

    syls = dask.compute(*syls)

    return syls


def main(csv_path,
         labelset,
         output_dir,
         spect_params_ref,
         audio_format,
         feature_group):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f'csv_path not found: {csv_path}'
        )

    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)

    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.exists() or not output_dir.is_dir():
        raise NotADirectoryError(
            f'output_dir not found, or not recognized as a directory:\n{output_dir}'
        )

    feature_list, _, _ = hvc.parse.extract._validate_feature_group_and_convert_to_list(feature_group)

    vak_df = pd.read_csv(csv_path)

    # pair annotation with audio file,
    # will use both to create `hvc.Syllable` instances from which we extract features
    annots = vak.annotation.from_df(vak_df)
    audio_paths = vak_df.audio_path.values
    audio_annot_map = vak.annotation.source_annot_map(audio_paths, annots)

    # also need a `spect_maker` to call `hvc.audiofileIO.make_syls`
    spect_params = hvc.parse.ref_spect_params.refs_dict[spect_params_ref]
    spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)

    # define this here so that `labelmap` and `feature_list` are in scope
    def ftrs_dict_from_syl(syl):
        ftrs = {}
        ftrs['label'] = labelmap[syl.label]  # string label -> integer

        for feature_name in feature_list:
            if feature_name in single_syl_features_switch_case_dict:
                ftr = single_syl_features_switch_case_dict[feature_name](syl)
            elif feature_name in multiple_syl_features_switch_case_dict:
                ftr = multiple_syl_features_switch_case_dict[feature_name](syl)
            else:
                raise ValueError(
                    f'could not determine which function to call for feature: {feature_name}'
                )
            # replace spaces w/underscores for DataFrame column names
            feature_name = feature_name.replace(' ', '_')
            if np.isscalar(ftr):
                ftrs[feature_name] = ftr
            else:
                for el_num, el in enumerate(ftr.tolist()):
                    ftr_el_name = f'{feature_name}_{el_num}'
                    ftrs[ftr_el_name] = el
        return ftrs

    ftrs_path_col = []  # gets added to dataset csv at end
    for audio_path, annot in tqdm(audio_annot_map.items()):
        raw_audio, samp_freq = vak.constants.AUDIO_FORMAT_FUNC_MAP[audio_format](audio_path)
        syls = make_syls(raw_audio=raw_audio,
                         samp_freq=samp_freq,
                         spect_maker=spect_maker,
                         labels=annot.seq.labels,
                         onsets_Hz=annot.seq.onsets_Hz,
                         offsets_Hz=annot.seq.offsets_Hz)

        ftr_dicts = []
        for syl in syls:
            ftr_dict = dask.delayed(ftrs_dict_from_syl)(syl)
            ftr_dicts.append(ftr_dict)

        records = dask.compute(*ftr_dicts)
        ftrs_df = pd.DataFrame.from_records(records)

        # make labels Series the first column
        label_series = ftrs_df.pop('label')
        ftrs_df.insert(0, 'labels', label_series)

        ftrs_path = output_dir / (Path(audio_path).stem + '.hvc.csv')
        ftrs_df.to_csv(ftrs_path, index=False)
        ftrs_path_col.append(ftrs_path)

    vak_df['features_path'] = ftrs_path_col
    vak_df.to_csv(output_dir / Path(csv_path.name), index=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path',
        type=Path,
        help='path to .csv representing a dataset, prepared by running `vak prep`'
    )
    parser.add_argument(
        '--labelset',
        help='set of labels used to annotate vocalizations'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='path to directory where output should be saved'
    )
    parser.add_argument(
        '--spect_params_ref',
        help='path to .csv representing a dataset, prepared by running `vak prep`',
        choices=('tachibana', 'koumura'),
        default='tachibana',
    )
    parser.add_argument(
        '--audio_format',
        help="format of audio files, one of: {'wav', 'cbin'}",
        choices=('wav', 'cbin')
    )
    parser.add_argument(
        '--feature_group',
        help="name of group of features for machine learning model, one of: {'svm', 'knn'}",
        choices=('svm', 'knn'),
        default='svm',
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(csv_path=args.csv_path,
         spect_params_ref=args.spect_params_ref,
         audio_format=args.audio_format,
         feature_group=args.feature_group,
         )
