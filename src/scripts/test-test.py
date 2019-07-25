from configparser import ConfigParser
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import tqdm

import vak

CONFIGS_DIR = Path('../../src/configs/')
BR_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BirdsongRecognition*ini')))
BR_CONFIGS = [str(config) for config in BR_CONFIGS
              if 'tmp' not in str(config)]

REDO_DIR = Path('../../data/BirdsongRecognition/redo_test')
REDO_DIR.mkdir()

timenow = datetime.now().strftime('%y%m%d_%H%M%S')
logger = logging.getLogger('learncurve')
if logging.getLevelName(logger.level) != 'INFO':
    logger.setLevel('INFO')

logfile_name = os.path.join(REDO_DIR,
                            'logfile_from_rerun_of_learncurve_test_' + timenow + '.log')
logger.addHandler(logging.FileHandler(logfile_name))
logger.info('Logging results to {}'.format(REDO_DIR))
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    for bird_num, config_ini in enumerate(BR_CONFIGS):
        config_obj = ConfigParser()
        config_obj.read(config_ini)

        data_config = vak.config.data.parse_data_config(config_obj, config_ini)
        train_config = vak.config.train.parse_train_config(config_obj, config_ini)
        net_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
        output_config = vak.config.output.parse_output_config(config_obj)
        results_dirname = output_config.results_dirname
        output_dir = REDO_DIR.joinpath(f'Bird{bird_num}')
        output_dir.mkdir()
        output_dir = str(output_dir)

        vak.core.learncurve.test(
            results_dirname=results_dirname,
            test_vds_path=train_config.test_vds_path,
            train_vds_path=train_config.train_vds_path,
            networks=net_config,
            train_set_durs=train_config.train_set_durs,
            num_replicates=train_config.num_replicates,
            output_dir=output_dir,
            normalize_spectrograms=train_config.normalize_spectrograms,
            save_transformed_data=False,
        )


if __name__ == '__main__':
    main()

