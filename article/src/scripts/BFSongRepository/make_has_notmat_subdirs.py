"""
data-munging script for analysis done with Bengalese Finch Song Repository
copy all .cbins with .not.mats into a sub-directory

run this before running bfsongrepo-test-predict.py
"""
import argparse
from pathlib import Path
import shutil

import tqdm


BIRDS = ['bl26lb16',
         'gy6or6',
         'or60yw70',
         'gr41rd51',
         ]


def main(bfsongrepo_root):
    bfsongrepo_root = Path(bfsongrepo_root).expanduser().resolve()

    bird_dirs = [bfsongrepo_root / bird for bird in BIRDS]
    bird_date_dirs = [
        subdir
        for bird_dir in bird_dirs
        for subdir in bird_dir.iterdir()
        if subdir.is_dir() and subdir.name.isnumeric()
    ]

    for bird_date_dir in bird_date_dirs:
        has_notmat = bird_date_dir.joinpath('has_notmat')
        has_notmat.mkdir(exist_ok=True)
        notmats_this_date_dir = sorted(list(bird_date_dir.glob('*.not.mat')))
        print(f'\ncopying annotated songs in {bird_date_dir} into sub-directory')
        for notmat in tqdm.tqdm(notmats_this_date_dir):
            shutil.copy(notmat, dst=has_notmat)
            cbin = notmat.parent.joinpath(
                Path(notmat.stem).stem
            )
            shutil.copy(cbin, dst=has_notmat)  # cbin_file, stem.stem removes .not.mat
            rec = notmat.parent.joinpath(
                Path(Path(notmat.stem).stem).stem + '.rec'
            )
            shutil.copy(rec, dst=has_notmat)
            tmp = notmat.parent.joinpath(
                Path(Path(notmat.stem).stem).stem + '.tmp'
            )
            shutil.copy(tmp, dst=has_notmat)


DEFAULT_BFSONGREPO_ROOT = '~/Documents/data/BFSongRepository/'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfsongrepo_root',
                        default=DEFAULT_BFSONGREPO_ROOT)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(bfsongrepo_root=args.bfsongrepo_root)
