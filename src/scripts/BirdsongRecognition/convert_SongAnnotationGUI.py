"""converts BirdsongRecognition dataset (https://figshare.com/articles/media/BirdsongRecognition/3470165)
to format recognized by SongAnnotationGUI: https://github.com/yardencsGitHub/BirdSongBout/tree/master/helpers/GUI
"""
import argparse
from collections import namedtuple
from datetime import datetime
from pathlib import Path
import shutil

import crowsetta
import numpy as np
from scipy.io import wavfile
from scipy.io import savemat
from tqdm import tqdm

BIRDS = [f'Bird{num}' for num in range(11)]


def main(birdsongrec_root, data_root):
    birdsongrec_root = Path(birdsongrec_root).expanduser().resolve()
    if not birdsongrec_root.exists():
        raise NotADirectoryError(
            f'birdsongrec_root not recognized as a directory: {birdsongrec_root}'
        )

    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        raise NotADirectoryError(
            f'data_root not recognized as a directory: {data_root}'
        )
    copyto_dir = data_root / 'annotation_converted'  # will copy generated .mat files here
    copyto_dir.mkdir(exist_ok=True)

    scribe = crowsetta.Transcriber(format='koumura')

    pbar = tqdm(BIRDS)
    for bird in pbar:
        pbar.set_description(f'bird={bird}')
        bird_dir = birdsongrec_root / bird
        annot_path = bird_dir / 'Annotation.xml'
        annots = scribe.from_file(annot_path)

        # determine unique labels here
        # so we can map them to discrete set of double values,
        # which is what SongAnnotationGui expects
        unique_labels = set(
            [lbl for annot in annots for lbl in annot.seq.labels.tolist()]
        )
        labelmap = {lbl: float(lbl_ind)
                    for lbl, lbl_ind in zip(sorted(unique_labels), range(len(unique_labels)))}

        # ---- first create annotation.mat file ------------------------------------------------------------------------
        # contains two variables, keys and elements
        # "keys" = list of filenames that SongAnnotationGUI matches to annotation "elements"
        # "elements" = Matlab structs each with fields filenum, segFileStartTimes, segFileEndTimes, segType, and fs
        keys, elements = [], []

        wave_dir = bird_dir / 'Wave'
        for filenum, annot in enumerate(annots):
            # note we convert audio_path.name to file name format expected by SongAnnotationGUI below
            key = annot.audio_path.name  # we use below to get 'fs', so assign to a variable
            keys.append(key)
            labels = np.array(
                [labelmap[lbl] for lbl in annot.seq.labels.tolist()]
            ).reshape(-1, 1)  # needs to be a col vector
            fs, _ = wavfile.read(str(wave_dir / key))  # we need fs for
            element = dict(
                filenum=str(filenum),
                segFileStartTimes=annot.seq.onsets_s.tolist(),
                segFileEndTimes=annot.seq.offsets_s.tolist(),
                segType=labels,
                fs=fs,
            )
            elements.append(element)

        # convert file names to the format expected by SongAnnotationGUI, 'birdname_nnnn_yyyy_mm_dd_HH_MM_SS'
        for key_ind in range(len(keys)):
            src = wave_dir / keys[key_ind]  # keys[key_ind] will be audio filename, e.g. '0.wav'
            dst_name = f"{bird}_{key_ind:04}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav"
            dst = wave_dir / dst_name
            shutil.move(src, dst)
            keys[key_ind] = dst_name  # notice we change filename in keys as well

        # save annotation, with changed filenames
        annot_dict = {'keys': np.array(keys, dtype=np.object), 'elements': elements}
        annot_fname = f'{bird}_annotation.mat'
        for dst in (bird_dir, copyto_dir, wave_dir):
            if dst == wave_dir:
                annot_fname = f'tweetynet-{annot_fname}'
            savemat(dst / annot_fname, annot_dict)

        # ---- second create template.mat file -------------------------------------------------------------------------
        # template file contains a struct 'template' with a single field 'wavs'
        # that is itself a struct array with field names as in WAV_STRUCT_FIELDS.
        # We create tuples in a loop that we use to build the numpy Structured Array below
        WAVS_STRUCT_FIELDS = ['filename', 'startTime', 'endTime', 'fs', 'wav', 'segType']
        WAVS_STRUCT_TUPLE = namedtuple('wav', field_names=WAVS_STRUCT_FIELDS)
        FAKE_WAV_CLIP_SIZE = (100,)  # has to be a fixed size, numpy constraint

        wavs_structs = []
        for key, element in zip(keys, elements):
            labels, onsets, offsets = element['segType'], element['segFileStartTimes'], element['segFileEndTimes']
            labels = np.squeeze(labels).tolist()
            for label, onset, offset in zip(labels, onsets, offsets):
                if label not in [wav.segType for wav in wavs_structs]:
                    fs, _ = wavfile.read(str(wave_dir / key))
                    fake_wav_clip = np.zeros(FAKE_WAV_CLIP_SIZE)
                    new_wav_struct = WAVS_STRUCT_TUPLE(
                        filename=key,
                        startTime=onset,
                        endTime=offset,
                        fs=fs,
                        wav=fake_wav_clip,
                        segType=label,
                    )
                    wavs_structs.append(new_wav_struct)
                if set([wav.segType for wav in wavs_structs]) == unique_labels:
                    break
            if set([wav.segType for wav in wavs_structs]) == unique_labels:
                break

        # need to create structured array dtype this **after** changing filenames,
        # because we need to know length of filenames for dtype
        FILENAME_SIZE = len(dst_name)  # will be fixed size for all filenames
        FIELD_DTYPES = (
            (np.unicode, FILENAME_SIZE),
            np.double,
            np.double,
            np.double,
            (np.float64, FAKE_WAV_CLIP_SIZE),
            np.float64,
        )
        WAVS_STRUCT_DTYPE = np.dtype([field_dtype for field_dtype in zip(WAVS_STRUCT_FIELDS, FIELD_DTYPES)])

        wavs_struct_array = np.array(
            wavs_structs,  # np.array will just ignore that these are **named** tuples
            dtype=WAVS_STRUCT_DTYPE
        )
        template_dict = {'templates': {'wavs': wavs_struct_array}}
        template_fname = f'{bird}_templates.mat'
        for dst in (bird_dir, copyto_dir, wave_dir):
            if dst == wave_dir:
                template_fname = f'tweetynet-{template_fname}'
            savemat(dst / template_fname, template_dict)


BIRDSONGREC_ROOT = Path('/Users/yardenc/Documents/Experiments/Koumura2016').expanduser().resolve()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--birdsongrec_root',
        default=BIRDSONGREC_ROOT,
        help='root of BirdsongRecognition dataset'
    )
    parser.add_argument(
        '--data_root',
        default='./data/BirdsongRecognition',
        help='root of "data" directory in `tweetynet` repository, where copies of output are saved'
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(birdsongrec_root=args.birdsongrec_root, data_root=args.data_root)
