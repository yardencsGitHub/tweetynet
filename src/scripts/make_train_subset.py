"""utilty script to take first TRAIN_DUR seconds of song (as determine from .not.mat files)
and move to a sub-directory called 'train_subset'. Execute in the directory where this should happen.
Used for experiments to measure accuracy when just using first TRAIN_DUR seconds of song to label 
the next n days of song.

run this before running make_has_notmat_subdirs.py
"""
import argparse
from glob import glob
import os
from pathlib import Path
import shutil

import crowsetta

TRAIN_DUR = 60
VAL_DUR = 80

ANNOT_EXT = '.not.mat'
VOC_FORMAT = 'notmat'

SUBSET_DIR = 'train_subset'

parser = argparse.ArgumentParser(description='Move subset of training data to a sub-directory.')
parser.add_argument('--labelset', dest='labelset', action='store', default=None,
                    help='set of labels each file in training subset should contain')
parser.add_argument('--train_dur', dest='train_dur', action='store', default=TRAIN_DUR,
                    help='duration of training set (in seconds)', type=int)
parser.add_argument('--subset_dir', dest='subset_dir', action='store', default=SUBSET_DIR,
                    help='name of directory in which to store subset for training data')


def main(train_dur=TRAIN_DUR,
         val_dur=VAL_DUR,
         annot_ext=ANNOT_EXT,
         voc_format=VOC_FORMAT,
         subset_dir=SUBSET_DIR,
         labelset=None):
    """makes training set of specified duration by taking subset of files
    in current directory and copying to a newly-created sub-directory
    """
    labelset = list(labelset)  # assumes single string
    annot_files = glob(f'*{annot_ext}')
    annot_files = sorted(annot_files)
    scribe = crowsetta.Transcriber(voc_format=voc_format)
    seqs = scribe.to_seq(annot_files)

    dur = 0
    seq_ctr = 0
    seqs_to_use = []
    total_dur = train_dur + val_dur
    while dur < total_dur:
        if seq_ctr > len(seqs):
            raise ValueError(
                f'ran out of annotation files before finding subset of duration {total_dur}'
            )

        if labelset:
            if not set(seqs[seq_ctr].labels).issubset(set(labelset)):
                seq_ctr += 1
                continue

        dur += seqs[seq_ctr].offsets_s[-1]
        seqs_to_use.append(seqs[seq_ctr])
        seq_ctr += 1

    os.makedirs(subset_dir)

    for seq in seqs_to_use:
        seq_stem = Path(seq.file).stem
        seq_files = glob(f'{seq_stem}*')
        for seq_file in seq_files:
            shutil.move(seq_file, subset_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(train_dur=args.train_dur, subset_dir=args.subset_dir, labelset=args.labelset)
