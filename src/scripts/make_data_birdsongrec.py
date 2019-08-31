#!/usr/bin/env python
# coding: utf-8
"""
(re)make data directory containing results obtained with BirdsongRecognition data repository

uses config.ini files from src/config/ to find the results of creating a learning curve for
song of each individual bird in directory, then copies sub-directory containing results of
measuring accuracy on training and test sets to the ./data/BirdsongRecognition directory
"""
from configparser import ConfigParser
from pathlib import Path
import shutil


REPO_ROOT = Path('~/Documents/repos/coding/birdsong/tweetynet/')
REPO_ROOT = REPO_ROOT.expanduser()
CONFIGS_DIR = REPO_ROOT.joinpath('src/configs/')
BR_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BirdsongRecognition*ini')))
BR_CONFIGS = [str(config) for config in BR_CONFIGS]
if not all([f'bird0{i}' in br_config for i, br_config in enumerate(BR_CONFIGS)]):
    raise ValueError(
        "could not find all config.ini files for BirdsongRecognition "
        "in consecutive order (i.e., 10 files with names that end in "
        "bird00.ini, bird01.ini, ... bird09.ini)"
    )
    
BR_DATA_ROOT = REPO_ROOT.joinpath('data/BirdsongRecognition')

# path to root of repository that contains results from running learncurve.train with each config.ini file
NEW_PARENT = '/media/art/HD-LCU3/tweetynet_paper/BirdsongRecognition'

# "roots" of paths in config.ini files that should be replaced with NEW_PARENT
OLD_PARENTS = [
    '/home/nickledave/Documents/data/BirdsongRecognition/vak',
    '~/Documents/data/birdsong/BirdsongRecognition/vak',
    '~/Documents/data/BirdsongRecognition/vak',
    '~/Documents/data/birdsong/vak',
]


def change_parent(path, new_parent=NEW_PARENT, old_parents=OLD_PARENTS):
    """changes parent directory of a path, given a list of possible
    'old' parents and the new parent that should replace any of those"""
    path = str(path)
    for old_parent in OLD_PARENTS:
        if old_parent in path:
            path = path.replace(old_parent, new_parent)

    assert new_parent in path, f'did not find parent to replace in {path}'
    path = Path(path)
    return path


def remove_subdirs(root_dir=BR_DATA_ROOT):
    """removes all sub-directories from a directory"""
    subdirs = [subdir for subdir in root_dir.iterdir() if subdir.is_dir()]
    for subdir in subdirs:
        shutil.rmtree(subdir)


def copy_test_dirs(br_configs=BR_CONFIGS, br_data_root=BR_DATA_ROOT,
                   new_parent=NEW_PARENT, old_parents=OLD_PARENTS):
    """copy test dir to root, using path from .ini file
    
    Parameters
    ----------
    old_parent : list
        of str
    new_parent : st
    br_configs : list
        of str, paths to config.ini files for BirdsongRecognition repository
    """
    config_obj = ConfigParser()
    for birdnum, config_ini in enumerate(br_configs):
        config_obj.read(config_ini)
        results_dirname = config_obj['OUTPUT']['results_dir_made_by_main_script']
        results_dirname = Path(results_dirname)
        
        src = results_dirname.joinpath('test')
        src = change_parent(src, new_parent, old_parents)
        dst = br_data_root.joinpath(f'Bird{birdnum}')
        if dst.exists():
            raise ValueError(f"can't copy to directory, already exists: {dst}")
        shutil.copytree(src, dst)


def main():
    remove_subdirs()
    copy_test_dirs()


if __name__ == '__main__':
    main()
