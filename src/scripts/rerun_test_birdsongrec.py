"""script that reproduces results of running vak.core.learncurve.test on models trained
using BirdsongRecognition data repository (https://figshare.com/articles/BirdsongRecognition/3470165)
"""
from configparser import ConfigParser
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import vak

CONFIGS_DIR = Path('src/configs/')
BR_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BirdsongRecognition*ini')))
BR_CONFIGS = [str(config) for config in BR_CONFIGS
              if 'tmp' not in str(config)]

BR_DATA_DIR = Path('data/BirdsongRecognition')

# path to root of repository that contains results from running learncurve.train with each config.ini file
RESULTS_REPO_ROOT = '/media/art/HD-LCU3/tweetynet_paper/BirdsongRecognition_copy_combined_backup'

# "roots" of paths in config.ini files that should be replaced with RESULTS_REPO_ROOT
ROOTS_TO_REPLACE = [
    '/home/nickledave/Documents/data/BirdsongRecognition/vak',
    '~/Documents/data/birdsong/BirdsongRecognition/vak',
    '~/Documents/data/BirdsongRecognition/vak',
    '~/Documents/data/birdsong/vak',
]

timenow = datetime.now().strftime('%y%m%d_%H%M%S')
logger = logging.getLogger('learncurve')
if logging.getLevelName(logger.level) != 'INFO':
    logger.setLevel('INFO')

logfile_name = os.path.join(BR_DATA_DIR,
                            'logfile_from_rerun_of_learncurve_test_' + timenow + '.log')
logger.addHandler(logging.FileHandler(logfile_name))
logger.info('Logging results to {}'.format(BR_DATA_DIR))
logger.addHandler(logging.StreamHandler(sys.stdout))


def change_path_to_root(path):
    for root in ROOTS_TO_REPLACE:
        if root in path:
            path = path.replace(root, RESULTS_REPO_ROOT)

    assert RESULTS_REPO_ROOT in path, f'did not find root to replace in {path}'
    return path


def main():
    for bird_num, config_ini in enumerate(BR_CONFIGS):
        config_obj = ConfigParser()
        config_obj.read(config_ini)

        # make paths required for test point to RESULTS_REPO_ROOT instead of original root path
        config_obj['TRAIN']['train_vds_path'] = change_path_to_root(config_obj['TRAIN']['train_vds_path'])
        config_obj['TRAIN']['val_vds_path'] = change_path_to_root(config_obj['TRAIN']['val_vds_path'])
        config_obj['TRAIN']['test_vds_path'] = change_path_to_root(config_obj['TRAIN']['test_vds_path'])
        config_obj['OUTPUT']['root_results_dir'] = change_path_to_root(
            config_obj['OUTPUT']['root_results_dir']
        )
        config_obj['OUTPUT']['results_dir_made_by_main_script'] = change_path_to_root(
            config_obj['OUTPUT']['results_dir_made_by_main_script']
        )

        output_dir = BR_DATA_DIR.joinpath(f'Bird{bird_num}')
        output_dir.mkdir()

        train_config = vak.config.train.parse_train_config(config_obj, config_ini)
        net_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
        output_config = vak.config.output.parse_output_config(config_obj)

        for vds_path in (train_config.train_vds_path, train_config.test_vds_path):
            vds = vak.Dataset.load(vds_path)
            new_root = os.path.join(RESULTS_REPO_ROOT, f'Bird{bird_num}')
            if not all(
                [new_root in voc.spect_path for voc in vds.voc_list]
            ):
                vds = vds.move_spects(new_root=new_root)
                vds.save(vds_path)

        vak.core.learncurve.test(
            results_dirname=output_config.results_dirname,
            test_vds_path=train_config.test_vds_path,
            train_vds_path=train_config.train_vds_path,
            networks=net_config,
            train_set_durs=train_config.train_set_durs,
            num_replicates=train_config.num_replicates,
            output_dir=str(output_dir),  # cuz it's a Path ... and we need it to be so below
            normalize_spectrograms=train_config.normalize_spectrograms,
            save_transformed_data=False,
        )

        # move contents of directory made by test up to parent
        test_dir = output_dir.joinpath('test')
        test_files = os.listdir(str(test_dir))
        for test_file in test_files:
            old_path = test_dir.joinpath(test_file).absolute()
            new_name = output_dir.joinpath(test_file)
            old_path.rename(new_name)
        test_dir.rmdir()


if __name__ == '__main__':
    main()

