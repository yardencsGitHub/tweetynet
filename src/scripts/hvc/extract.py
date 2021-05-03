import argparse
from collections import defaultdict
from pathlib import Path

import hvc
from hvc.features.feature_dicts import (
    single_syl_features_switch_case_dict,
    multiple_syl_features_switch_case_dict
)
import numpy as np
import pandas as pd
import vak
import vak.converters


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

    ftrs_path_col = []  # gets added to dataset csv at end
    for audio_path, annot in audio_annot_map.items():
        raw_audio, samp_freq = vak.constants.AUDIO_FORMAT_FUNC_MAP[audio_format](audio_path)
        syls = hvc.audiofileIO.make_syls(raw_audio=raw_audio,
                                         samp_freq=samp_freq,
                                         spect_maker=spect_maker,
                                         labels=annot.seq.labels,
                                         onsets_Hz=annot.seq.onsets_Hz,
                                         offsets_Hz=annot.seq.offsets_Hz)
        ftrs_dict = defaultdict(list)
        for syl in syls:
            ftrs_dict['label'].append(
                labelmap[syl.label]  # string label -> integer
            )
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
                if np.isscalar():
                    ftrs_dict[feature_name].append(ftr)
                else:
                    for el_num, el in enumerate(ftr.tolist()):
                        ftrs_dict[f'{feature_name}_{el_num}'].append(el)
        ftrs_df = pd.DataFrame.from_records(ftrs_dict)
        ftrs_path = Path(audio_path.parent) / Path(audio_path.stem) + '.hvc.ftrs'
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
        default='koumura',
    )
    parser.add_argument(
        '--audio_format',
        help="format of audio files, one of: {'wav', 'cbin'}",
        choices=('wav', 'cbin')
    )
    parser.add_argument(
        '--feature_group',
        help="name of group of features for machine learning model, one of: {'svm', 'knn'}",
        choices=('svm', 'knn')
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
