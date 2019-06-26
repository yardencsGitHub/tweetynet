"""utility functions used to munge data for article"""
import pickle
from configparser import ConfigParser

import joblib
import pandas as pd


FIELDS = [
    'animal_ID',
    'train_set_dur',
    'replicate',
    'net_name',
    'learning_rate',
    'time_bins',
    'frame_error_train',
    'frame_error_test',
    'syllable_error_train',
    'syllable_error_test',
]


def make_df(config_files, test_dirs, net_name='TweetyNet', csv_fname=None):
    """make Pandas dataframe of results from running vak.core.learning_curve,
    given list of config.ini files and lists of directories with results from
    vak.core.learning_curve.test

    Parameters
    ----------
    config_files : list
        of pathlib.Path objects; paths to config.ini files used to run vak.core.learning_curve
    test_dirs : list
        of pathlib.Path objects; directories output by vak.core.learning_curve.test.
        The parent directory of the file 'y_pres_and_err_for_train_and_test' is assumed to be
        the ID of the bird from which results were generated.
    net_name : str
        name of network that is also the name of the section in the config.ini file
        that defines its hyperparameters. Default is 'TweetyNet'.
    csv_fname : str
        filename used to save dataframe as a csv. Default is None, in which case no .csv is
        saved.

    Returns
    -------
    results_df : Pandas.Dataframe
        with the following columns:
            animal_ID : str
                identity of animal from which vocalizations were recorded. Assumed to be name of test_dir that contains
                results. e.g. "data/BirdsongRecognition/Bird1/" would result in Bird1 being the animal_ID.
            train_set_dur : float
                duration of training data set. Determined from config.ini file.
            replicate : int
                training replicate, e.g. net number 1 of 5 trained with random subset
                with total duration `train_set_dur` from a larger set of training data.
                Determined from train_err file.
            learning_rate : float
                hyperparameter, update rate for weights in neural network. Determined
                from config.ini file.
            time_bins : int
                hyperparameter, number of time bins from spectrogram in windows shown to
                network. Determined from config.ini file.
            frame_error : float
                frame error rate, i.e. number of frames / time bins incorrectly classified
            syllable_error : float
                syllable error rate, i.e. Levenstein distance between original sequence of labels and sequence
                of labels recovered from predicted
    Notes
    -----
    This function assumes that config_files and test_dirs are in the same order, i.e. that
    config_files[0] is the config.ini file that produced the results in test_dirs[0].
    You can ensure this is the case by applying the `sorted` function to the lists, if
    you used the same name (e.g. "Bird1") in both the config.ini file and the test directory
    (or one of its parent directories, e.g. "Bird1/learning_curve/test/").
    """
    df_dict = {field: [] for field in FIELDS}

    for config_file, test_dir in zip(config_files, test_dirs):
        # ------------ get what we need from config.ini ----------------------------------------------------------------
        config = ConfigParser()
        config.read(config_file)

        # ------------ need train set durations to pair with error values ----------------------------------------------
        train_set_durs = [int(element)
                          for element in
                          config['TRAIN']['train_set_durs'].split(',')]
        # ------------ then need hyperparameters -----------------------------------------------------------------------
        time_bins = config[net_name]['time_bins']
        learning_rate = config[net_name]['learning_rate']

        # ------------ now unpack everything from test_dir -------------------------------------------------------------
        animal_ID = test_dir.stem  # assume stem (i.e. last part of directory) is animal ID #
        train_err = list(test_dir.glob('train_err'))
        assert len(train_err) == 1, f"did not find only one train_err file: {train_err}"
        train_err = train_err[0]
        with open(train_err, 'rb') as f:
            frame_err_train = pickle.load(f)

        test_err = list(test_dir.glob('test_err'))
        assert len(test_err) == 1, f"did not find only one test_err file: {test_err}"
        test_err = test_err[0]
        with open(test_err, 'rb') as f:
            frame_err_test = pickle.load(f)

        preds_and_err = list(test_dir.glob('y_preds_and_err_for_train_and_test'))
        assert len(preds_and_err) == 1, f"did not find only one preds_and_err file: {preds_and_err}"
        preds_and_err = preds_and_err[0]
        preds_and_err = joblib.load(preds_and_err)
        syl_err_train = preds_and_err['train_syl_err_rate']
        syl_err_test = preds_and_err['test_syl_err_rate']

        err_zip = zip(train_set_durs, frame_err_train, frame_err_test, syl_err_train, syl_err_test)
        for train_set_dur, fe_tr_row, fe_te_row, se_tr_row, se_test_row in err_zip:
            err_rows = [fe_tr_row, fe_te_row, se_tr_row, se_test_row]
            row_lengths = [len(row) for row in err_rows]
            assert len(set(row_lengths)) == 1, f"found different lengths of rows in error arrays: {row_lengths}"
            for replicate, (fe_tr, fe_te, se_tr, se_te) in enumerate(zip(*err_rows)):
                df_dict['animal_ID'].append(animal_ID)
                df_dict['train_set_dur'].append(train_set_dur)
                df_dict['replicate'].append(replicate)
                df_dict['net_name'].append(net_name)
                df_dict['time_bins'].append(time_bins)
                df_dict['learning_rate'].append(learning_rate)
                df_dict['frame_error_train'].append(fe_tr)
                df_dict['frame_error_test'].append(fe_te)
                df_dict['syllable_error_train'].append(se_tr)
                df_dict['syllable_error_test'].append(se_te)

    df = pd.DataFrame.from_dict(df_dict)

    if csv_fname:
        df.to_csv(csv_fname)

    return df


def agg_df(df, train_set_durs):
    """filters dataframe of results by specified training set durations, then
    groups by animal ID and training set duration, and for each group computes
    the mean and standard deviation, i.e. aggregate statistics.

    The index of each row becomes one animal ID, and this function makes mean
    and standard deviations into columns, as well as the training set duration.
    This way the dataframe is in "long" form (where "animal ID" and "training set
    durations" are the conditions) for use with Seaborn.

    Parameters
    ----------
    df : pandas.Dataframe
    train_set_durs :

    Returns
    -------
    df_agg : pandas.Dataframe
    """
    df = df[df.train_set_dur.isin(train_set_durs)]
    df_agg = df.groupby(['animal_ID', 'train_set_dur']).agg({'frame_error_train': ['mean', 'std'],
                                                             'frame_error_test': ['mean', 'std'],
                                                             'syllable_error_train': ['mean', 'std'],
                                                             'syllable_error_test': ['mean', 'std']})
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index(level='train_set_dur')
    return df_agg
