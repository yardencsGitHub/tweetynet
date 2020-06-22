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

TRAIN_DUR = 180
VAL_DUR = 80

ANNOT_EXT = '.not.mat'
ANNOT_FORMAT = 'notmat'

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
         annot_format=ANNOT_FORMAT,
         subset_dir=SUBSET_DIR,
         labelset=None):
    """makes training set of specified duration by taking subset of files
    in current directory and copying to a newly-created sub-directory
    """
    labelset = list(labelset)  # assumes single string
    annot_files = glob(f'*{annot_ext}')
    annot_files = sorted(annot_files)
    scribe = crowsetta.Transcriber(annot_format=annot_format)
    annots = scribe.from_file(annot_files)

    dur = 0
    annot_ctr = 0
    annots_to_use = []
    total_dur = train_dur + val_dur
    while dur < total_dur:
        if annot_ctr > len(annots):
            raise ValueError(
                f'ran out of annotation files before finding subset of duration {total_dur}'
            )

        if labelset:
            if not set(annots[annot_ctr].seq.labels).issubset(set(labelset)):
                annot_ctr += 1
                continue

        dur += annots[annot_ctr].seq.offsets_s[-1]
        annots_to_use.append(annots[annot_ctr])
        annot_ctr += 1

    os.makedirs(subset_dir)

    for annot in annots_to_use:
        annot_stem = Path(annot.annot_file).name.split('.')[0]
        files_this_annot = glob(f'{annot_stem}*')
        for file in files_this_annot:
            shutil.move(file, subset_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(train_dur=args.train_dur, subset_dir=args.subset_dir, labelset=args.labelset)
