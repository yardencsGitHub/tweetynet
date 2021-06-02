from pathlib import Path

import dask
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
            freq_cutoffs=None,
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


# this is the output of calling
# hvc.parse.extract._validate_feature_group_and_convert_to_list('svm')
# but re-ordered to make sure features are in same order as they are
# returned by the original matlab script written by Tachibana
FEATURE_LIST = [
    'mean spectrum',
    'mean delta spectrum',
    'mean cepstrum',
    'mean delta cepstrum',
    'duration',
    'mean spectral centroid',
    'mean spectral spread',
    'mean spectral skewness',
    'mean spectral kurtosis',
    'mean spectral flatness',
    'mean spectral slope',
    'mean pitch',
    'mean pitch goodness',
    'mean amplitude',
    'zero crossings',
    'mean delta spectral centroid',
    'mean delta spectral spread',
    'mean delta spectral skewness',
    'mean delta spectral kurtosis',
    'mean delta spectral flatness',
    'mean delta spectral slope',
    'mean delta pitch',
    'mean delta pitch goodness',
    'mean delta amplitude'
]


def extract(csv_path,
            labelset,
            features_dst,
            csv_dst,
            spect_maker,
            audio_format,
            feature_list=FEATURE_LIST):
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f'csv_path not found: {csv_path}'
        )

    labelset = vak.converters.labelset_to_set(labelset)
    labelmap = vak.labels.to_map(labelset, map_unlabeled=False)

    features_dst = Path(features_dst).expanduser().resolve()
    if not features_dst.exists() or not features_dst.is_dir():
        raise NotADirectoryError(
            f'features_dst not found, or not recognized as a directory:\n{features_dst}'
        )

    csv_dst = Path(csv_dst).expanduser().resolve()
    if not csv_dst.exists() or not csv_dst.is_dir():
        raise NotADirectoryError(
            f'csv_dst not found, or not recognized as a directory:\n{csv_dst}'
        )

    vak_df = pd.read_csv(csv_path)

    # pair annotation with audio file,
    # will use both to create `hvc.Syllable` instances from which we extract features
    annots = vak.annotation.from_df(vak_df)
    audio_paths = vak_df.audio_path.values
    audio_annot_map = vak.annotation.source_annot_map(audio_paths, annots)

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

        # .not.mat annots do not have 'onsets_Hz' and 'offsets_Hz' attributes
        # so we compute dynamically
        onset_inds = np.round(annot.seq.onsets_s * samp_freq).astype(int)
        offset_inds = np.round(annot.seq.offsets_s * samp_freq).astype(int)

        syls = make_syls(raw_audio=raw_audio,
                         samp_freq=samp_freq,
                         spect_maker=spect_maker,
                         labels=annot.seq.labels,
                         onsets_Hz=onset_inds,
                         offsets_Hz=offset_inds)

        ftr_dicts = []
        for syl in syls:
            ftr_dict = dask.delayed(ftrs_dict_from_syl)(syl)
            ftr_dicts.append(ftr_dict)

        records = dask.compute(*ftr_dicts)
        ftrs_df = pd.DataFrame.from_records(records)

        # make labels Series the first column
        label_series = ftrs_df.pop('label')
        ftrs_df.insert(0, 'labels', label_series)

        ftrs_path = features_dst / (Path(audio_path).stem + '.hvc.csv')
        ftrs_df.to_csv(ftrs_path, index=False)
        ftrs_path_col.append(ftrs_path)

    vak_df['features_path'] = ftrs_path_col
    vak_df.to_csv(csv_dst / Path(csv_path.name), index=False)
