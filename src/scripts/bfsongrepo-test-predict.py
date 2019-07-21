from configparser import ConfigParser
import os
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import tqdm

import vak
from vak.utils.data import reshape_data_for_batching

BIRDS = ['bl26lb16', 'gy6or6', 'or60yw70', 'gr41rd51']

HERE = Path(__file__).parent
CONFIGS_DIR = HERE.joinpath('../configs/')
DATA_DIR = HERE.joinpath('../../data/BFSongRecognition')
BF_CONFIGS = sorted(list(CONFIGS_DIR.glob('*BFSongRepository*ini')))

configs_by_bird = {
    bird: [bf_config for bf_config in BF_CONFIGS if bird in str(bf_config)][0]
    for bird in BIRDS
}

BFSongRepo = Path('~/Documents/data/BFSongRepository/').expanduser()

all_notmats = list(BFSongRepo.glob('*/*/*.not.mat'))
bird_date_dirs = set([notmat.parents[0] for notmat in all_notmats])


# copy all .cbins with .not.mats into a sub-directory

# In[10]:


# for bird_date_dir in bird_date_dirs:
#     has_notmat = bird_date_dir.joinpath('has_notmat')
#     has_notmat.mkdir(exist_ok=True)
#     notmats_this_date_dir = sorted(list(bird_date_dir.glob('*.not.mat')))
#     print(f'\ncopying annotated songs in {bird_date_dir} into sub-directory')
#     for notmat in tqdm.tqdm(notmats_this_date_dir):
#         shutil.copy(notmat, dst=has_notmat)
#         cbin = notmat.parent.joinpath(
#             Path(notmat.stem).stem
#         )
#         shutil.copy(cbin, dst=has_notmat)  # cbin_file, stem.stem removes .not.mat
#         rec = notmat.parent.joinpath(
#             Path(Path(notmat.stem).stem).stem + '.rec'
#         )
#         shutil.copy(rec, dst=has_notmat)
#         tmp = notmat.parent.joinpath(
#             Path(Path(notmat.stem).stem).stem + '.tmp'
#         )
#         shutil.copy(tmp, dst=has_notmat)


# get dirs to predict for each bird

# In[4]:

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


NETWORKS = vak.network._load()


def main():
    for bird in BIRDS:
        print(f'predicting segments and labels for bird: {bird}')
        output_dir_this_bird = DATA_DIR.joinpath(bird)
        output_dir_this_bird.mkdir()
        print(f'saving output in: {output_dir_this_bird}')
        output_dir_this_bird_vds = output_dir_this_bird.joinpath('vds')
        output_dir_this_bird_vds.mkdir()
        output_dir_this_bird_json = output_dir_this_bird.joinpath('json')
        output_dir_this_bird_json.mkdir()

        config_ini = configs_by_bird[bird]
        config_obj = ConfigParser()
        config_obj.read(config_ini)

        data_config = vak.config.data.parse_data_config(config_obj, config_ini)
        train_config = vak.config.train.parse_train_config(config_obj, config_ini)
        net_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
        tweetynet_name, tweetynet_config = list(net_config.items())[0]

        results_dir = config_obj['OUTPUT']['results_dir_made_by_main_script']
        checkpoint_path = str(Path(results_dir).joinpath('TweetyNet'))
        spect_scaler_path = str(Path(results_dir).joinpath('spect_scaler'))
        spect_scaler = joblib.load(spect_scaler_path)

        # TODO: fix path
        print(f'\tgetting labelmap from {train_config.train_vds_path}')
        train_vds = vak.dataset.VocalizationDataset.load(train_config.train_vds_path)
        train_vds = train_vds.load_spects()
        labelmap = train_vds.labelmap

        bird_dirs_predict = dirs_to_predict[bird]    
        for dir_to_predict in bird_dirs_predict:
            stem = f'{dir_to_predict.parents[0].name}.{dir_to_predict.name}'

            X_train = train_vds.spects_list()
            X_train = np.concatenate(X_train, axis=1)
            Y_train = train_vds.lbl_tb_list()
            Y_train = np.concatenate(Y_train)
            # transpose so rows are time bins
            X_train = X_train.T
            freq_bins = X_train.shape[-1]  # number of columns
            X_train = spect_scaler.transform(X_train)


            test_vds = vak.dataset.prep(str(dir_to_predict),
                                        annot_format='notmat',
                                        labelset=data_config.labelset,
                                        output_dir=dir_to_predict,
                                        save_vds=False,
                                        return_vds=True,
                                        return_path=False,
                                        audio_format='cbin',
                                        spect_params=sp_nt)

            n_classes = len(labelmap)
            tweetynet_config_dict = tweetynet_config._asdict()
            tweetynet_config_dict['n_syllables'] = n_classes
            if 'freq_bins' in tweetynet_config_dict:
                tweetynet_config_dict['freq_bins'] = freq_bins

            X_test = test_vds.spects_list()
            X_test = np.concatenate(X_test, axis=1)
            # transpose so rows are time bins
            X_test = X_test.T
            Y_test = test_vds.lbl_tb_list()
            Y_test = np.concatenate(Y_test)
            X_test = spect_scaler(X_test)

            (X_train,
             _,
             num_batches_train) = reshape_data_for_batching(X_train,
                                                            tweetynet_config.batch_size,
                                                            tweetynet_config.time_bins,
                                                            Y_train)

            # Notice we don't reshape Y_test
            (X_test,
             _,
             num_batches_test) = reshape_data_for_batching(X_test,
                                                           tweetynet_config.batch_size,
                                                           tweetynet_config.time_bins,
                                                           Y_test)
            
            
            print("running test on data from {dir_to_predict}")
            (Y_pred_train,
             Y_pred_test,
             Y_pred_train_labels,
             Y_pred_test_labels,
             train_err,
             train_lev,
             train_syl_err_rate,
             test_err,
             test_lev,
             test_syl_err_rate) = vak.core.learncurve.test_one_model(tweetynet_name,
                                                                     tweetynet_config_dict,
                                                                     NETWORKS,
                                                                     n_classes,
                                                                     labelmap,
                                                                     checkpoint_path,
                                                                     X_train,
                                                           
                                                                     Y_train,
                                                                     num_batches_train,
                                                                     X_test,
                                                                     Y_test,
                                                                     num_batches_test)

            print(f'error on training set: {train_err}')
            print(f'Levenstein distance on training set: {train_lev}')
            print(f'syllable error rate on training set: {train_syl_err_rate}')
            print(f'error on test set: {test_err}')
            print(f'Levenstein distance on test set: {test_lev}')
            print(f'syllable error rate on test set: {test_syl_err_rate}')
            
            err_dict = {
                'train_err': float(train_err),
                'train_lev': int(train_lev),
                'train_syl_err_rate': float(train_syl_err_rate),
                'test_err': float(test_err),
                'test_lev': int(test_lev),
                'test_syl_err_rate': float(test_syl_err_rate),
            }
            test_json_path = str(output_dir_this_bird_json.joinpath(f'{stem}.test.json'))
            print(f'saving test results in {test_json_path}')
            with open(test_json_path, 'w') as fp:
                json.dump(err_dict, fp)
            test_vds = test_vds.clear_spects()
            test_vds_fname = str(output_dir_this_bird_vds.joinpath(
                f'{stem}.test.vds.json'
            ))
            print(f'saving test Dataset as {test_vds_fname}')
            test_vds.save(test_vds_fname)
            
            predict_vds_fname = str(output_dir_this_bird_vds.joinpath(
                f'{stem}.predict.vds.json'
            ))
            print(f'\tmaking dataset for predictions from {dir_to_predict}')
            predict_vds = vak.dataset.prep(str(dir_to_predict),
                                           audio_format='cbin',
                                           spect_params=sp_nt,
                                           return_vds=True,
                                           return_path=False)
            predict_vds = predict_vds.clear_spects()
            predict_vds.save(json_fname=predict_vds_fname)

            print(f'\trunning vak.core.predict on {dir_to_predict}')
            vak.core.predict(
                predict_vds_path=predict_vds_fname,
                checkpoint_path=checkpoint_path,
                networks=net_config,
                labelmap=labelmap,
                spect_scaler_path=spect_scaler_path)


if __name__ == '__main__':
    main()



