import argparse
from pathlib import Path
import warnings

import crowsetta
import evfuncs
import numpy as np
import pandas as pd
import pyprojroot
import soundfile
from tqdm import tqdm


PROJ_ROOT = pyprojroot.here()


def get_seg_feature_vals(audio_data_dir,
                         annot_data_dir):
    scribe = crowsetta.Transcriber(format='yarden')
    annot_file = sorted(
        annot_data_dir.glob('*annotation*.mat')
    )[-1]
    # just use most recent annotation file according to `sorted`; close enough for our purposes
    annots = scribe.from_file(annot_file)

    amplitudes = []
    segment_durs = []
    silent_intervals = []

    pbar = tqdm(annots)
    for annot in pbar:
        audio_name = Path(annot.audio_path).name
        pbar.set_description(f'processing: {audio_name}')
        audio_path = audio_data_dir / audio_name
        if not audio_path.exists():
            raise FileNotFoundError(
                f'did not find audio file: {audio_path}'
            )

        # unpack since we use multiple times
        onsets_s = annot.seq.onsets_s
        offsets_s = annot.seq.offsets_s
        # get threshold values;
        # note range for this audio is ~float(-1.0, 1.0) so thresholds will
        # be much smaller than the binary format with integer values used for .cbin audio files
        rawsong, samp_freq = soundfile.read(audio_path)
        smooth = evfuncs.smooth_data(rawsong, samp_freq)
        onsets_sample_inds = np.round(onsets_s * samp_freq).astype(int)
        offsets_sample_inds = np.round(offsets_s * samp_freq).astype(int)
        # what is the amplitude at every onset and offset in ground truth annotations?
        # we will use some estimate of those values as a "least bad" threshold
        len_smooth = smooth.shape[0]
        if any([np.any(onsets_sample_inds > len_smooth), np.any(offsets_sample_inds > len_smooth)]):
            warnings.warn(
                f'found onset or offset outside length of audio array: {audio_name}. Removing those on/offsets'
            )
            onsets_sample_inds = onsets_sample_inds[onsets_sample_inds < len_smooth]
            offsets_sample_inds = offsets_sample_inds[offsets_sample_inds < len_smooth]
        amplitude_vals = np.array(
            [*smooth[onsets_sample_inds].tolist(), *smooth[offsets_sample_inds].tolist()]
        )
        amplitudes.append(amplitude_vals)

        # update min seg dur and silent dur
        silent_gap_durs = onsets_s[1:] - offsets_s[:-1]  # duration of silent gaps
        silent_intervals.append(silent_gap_durs)

        syl_durs = offsets_s - onsets_s
        segment_durs.append(syl_durs)

    amplitudes = np.concatenate(amplitudes)
    silent_intervals = np.concatenate(silent_intervals)
    segment_durs = np.concatenate(segment_durs)

    return {
        'amplitude': amplitudes,
        'segment_dur': segment_durs,
        'silent_interval': silent_intervals
    }


CSV_DST = Path('./results/Canaries/seg_stats/')


def main(canary_data_root,
         canary_ids=('llb11', 'llb16', 'llb3'),
         audio_data_dirname='annotated',
         annot_data_dirnames=('annotation', 'annotation_files'),
         csv_dst=CSV_DST,
         ):
    canary_data_root = Path(canary_data_root).expanduser()
    if not canary_data_root.exists():
        raise NotADirectoryError(
            f'did not find directory specified as `canary_data_root`: {canary_data_root}'
        )

    csv_dst = Path(csv_dst).expanduser()
    if not csv_dst.exists():
        raise NotADirectoryError(
            f'did not find directory specified as `csv_dst`: {csv_dst}'
        )

    todo = []
    for canary_id in canary_ids:
        canary_audio_data_dir = canary_data_root / canary_id / audio_data_dirname
        if not canary_audio_data_dir.exists():
            raise NotADirectoryError(
                f'did not find directory for canary audio data`: {canary_audio_data_dir}'
            )

        canary_annot_data_dir = None
        for annot_data_dirname in annot_data_dirnames:
            canary_annot_data_dir = canary_data_root / canary_id / annot_data_dirname
            if canary_annot_data_dir.exists():
                break
        if not canary_annot_data_dir.exists():
            raise NotADirectoryError(
                f'unable to find annotation data for: {canary_data_root / canary_id}. '
                f'Tried the following directory names: {annot_data_dirnames}'
            )

        todo.append(
            (canary_id, canary_audio_data_dir, canary_annot_data_dir)
        )

    SEG_FEATURES = ('amplitude',
                    'segment_dur',
                    'silent_interval',
                    )

    for todo_num, (canary_id, audio_data_dir, annot_data_dir) in enumerate(todo):
        seg_feature_vals = get_seg_feature_vals(
            audio_data_dir,
            annot_data_dir,
        )
        for seg_feature in SEG_FEATURES:
            records = [
                {seg_feature: seg_feature_val}
                for seg_feature_val in seg_feature_vals[seg_feature]
            ]
            df_seg_feature = pd.DataFrame.from_records(records)
            df_seg_feature['animal_id'] = canary_id
            csv_path = csv_dst / f'{seg_feature}.csv'
            if todo_num == 0:
                # saves new .csv, overwriting any from a previous run of this script
                pass
            else:
                prev_df = pd.read_csv(csv_path)
                df_seg_feature = pd.concat((prev_df, df_seg_feature))
            df_seg_feature.to_csv(csv_path, index=False)

        # avoid keeping in memory
        del seg_feature_vals
        del df_seg_feature
        if 'prev_df' in locals():
            del prev_df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--canary-data-root',
        default='~/Documents/data/birdsong/canary'
    )
    parser.add_argument(
        '--canary-ids',
        nargs='+',
        default=('llb11', 'llb16', 'llb3')
    )
    parser.add_argument(
        '--audio-data-dirname',
        default='annotated'
    )
    parser.add_argument(
        '--annot-data-dirnames',
        nargs='+',
        default=('annotation', 'annotation_files')
    )
    parser.add_argument('--csv-dst',
                        help=('path to destination of .csv files '
                              f'that will be saved by this script, default is: {CSV_DST}'),
                        default=CSV_DST)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(
        canary_data_root=args.canary_data_root,
        canary_ids=args.canary_ids,
        audio_data_dirname=args.audio_data_dirname,
        annot_data_dirnames=args.annot_data_dirnames,
        csv_dst=args.csv_dst,
    )
