from configparser import ConfigParser
import os
from pathlib import Path
import shutil

import vak


BIRDS = ['bl26lb16', 'gy6or6', 'or60yw70', 'gr41rd51']

# get configs so we can get
CONFIGS_DIR = Path('../../src/configs/')
BF_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BFSongRepository*ini')))


configs_by_bird = {
    bird: [bf_config for bf_config in BF_CONFIGS if bird in str(bf_config)][0]
    for bird in BIRDS
}


BFSongRepo = Path('~/Documents/data/birdsong/BFSongRepository/')

all_notmats = list(BFSongRepo.glob('*/*/*.not.mat'))
bird_date_dirs = set([notmat.parents[0] for notmat in all_notmats])


# having already copied any labeled song into 'has_notmat' subdirectory
dirs_to_predict = {}
for bird in BIRDS:
    these = [
        bird_date_dir for bird_date_dir in bird_date_dirs
        if bird in str(bird_date_dir)
    ]
    these = [path.joinpath('has_notmat')
             for path in these]
    dirs_to_predict[bird] = these


spect_params = {'fft_size': 512,
                'step_size': 62,
                'freq_cutoffs': [500, 10000],
                'thresh': 6.25,
                'transform_type': 'log_spect'}
sp_nt = vak.config.spectrogram.SpectConfig(**spect_params)

for bird in BIRDS:
    config_ini = configs_by_bird[bird]
    config_obj = ConfigParser()
    config_obj.read(config_ini)

    train_config = vak.config.train.parse_train_config(config_obj, config_file)
    net_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)

    results_dir = config_obj['OUTPUT']['results_dir_made_by_main_script']
    training_records_dir = list(
        Path(a_results_dir).joinpath('train').glob(
            'records_for_training_set*'))[0]
    checkpoint_path = str(Path(training_records_dir).joinpath(
        'TweetyNet', 'checkpoints'))
    spect_scaler_path = list(
        Path(training_records_dir).glob('spect_scaler_*'))[0]
    spect_scaler_path = str(spect_scaler)

    # TODO: fix path
    train_vds = vak.dataset.VocalizationDataset.load(train_config.train_vds_path)
    train_vds = train_vds.load_spects()
    labelmap = train_vds.labelmap
    
    bird_dirs_predict = dirs_to_predict[bird]    
    for dir_to_predict in bird_dirs_predict:
        vds_fname = str(birds_dirs_predict.joinpath('predict.vds.json'))
        vak.dataset.prep(data_dir,
                         audio_format='cbin',
                         spect_params=sp_nt,
                         vds_fname=vds_fname,
                         return_vds=False,)
        vak.cli.predict(
            predict_vds_path=vds_fname,
            checkpoint_path=checkpoint_path,
            networks=net_config,
            labelmap=labelmap,
            spect_scaler_path=spect_scaler_path,
            save_predict_vds=True,
        )

