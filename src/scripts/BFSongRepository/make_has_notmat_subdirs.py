"""
data-munging script for analysis done with Bengalese Finch Song Repository
copy all .cbins with .not.mats into a sub-directory

run this before running bfsongrepo-test-predict.py
"""
from pathlib import Path
import shutil

import tqdm


BIRDS = ['bl26lb16',
         'gy6or6',
         'or60yw70',
         'gr41rd51',
         ]

HERE = Path(__file__).parent
CONFIGS_DIR = HERE.joinpath('../configs/')
DATA_DIR = HERE.joinpath('../../data/BFSongRepository')
BF_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BFSongRepository*ini')))

configs_by_bird = {
    bird: [bf_config for bf_config in BF_CONFIGS if bird in str(bf_config)][0]
    for bird in BIRDS
}

BFSongRepo = Path('~/Documents/data/BFSongRepository/').expanduser()

all_notmats = list(BFSongRepo.glob('*/*/*.not.mat'))
bird_date_dirs = set([notmat.parents[0] for notmat in all_notmats])


def main():
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


if __name__ == '__main__':
    main()
