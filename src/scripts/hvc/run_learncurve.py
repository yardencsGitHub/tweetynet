import argparse
import configparser
from pathlib import Path

import evfuncs
import hvc
import numpy as np
import pandas as pd
import pyprojroot
from tqdm import tqdm

import article.hvc


def rerun_learncurve(animal_id_root,
                     segment_params):

    animal_id = animal_id_root.name

    article.hvc.resegment(prep_csv_path,
                          seg_params,
                          annot_dst=f'./results/hvc/annot/{animal_id}',
                          csv_dst=f'./results/hvc/csv/{animal_id}',
                          split='test')

    # also need a `spect_maker` to call `hvc.audiofileIO.make_syls`
    spect_params = hvc.parse.ref_spect_params.refs_dict[spect_params_ref]
    spect_params['window'] = 'Hann'
    spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)

    article.hvc.extract(csv_path,
                        labelset,
                        features_dst,
                        csv_dst,
                        spect_maker,
                        audio_format,
                        FEATURE_LIST)

    article.hvc.extract(csv_path='results/hvc/csv/gy6or6/032212_prep_210423_094825.resgment.csv',
                        labelset='-',
                        features_dst='results/hvc/features/gy6or6/resegment/',
                        csv_dst=csv_dst,
                        spect_maker=spect_maker,
                        audio_format=audio_format,
                        feature_list=article.hvc.extract.FEATURE_LIST)

    learncurve_results_root = Path('results/Bengalese_Finches/learncurve/gy6or6/results_210509_010443/')
    prep_csvs = vak.core.learncurve.train_dur_csv_paths._dict_from_dir(learncurve_results_root)
    TRAINSET_DUR = 600
    # prep_csvs = sorted(learncurve_results_root.glob('train_dur*/replicate*/*prep*csv'))
    #### NOTICE I am using one of the duration = 600
    learncurve_csv_path = prep_csvs[TRAINSET_DUR][0]

    clf = fit(extract_csv_path,
              learncurve_csv_path)

    clf_dst = Path('./results/hvc/models/gy6or6')
    save_clf(clf, clf_dst)

    csv_root = Path('results/hvc/csv/gy6or6')
    prep_csv_path = csv_root / '032212_prep_210423_094825.csv'
    resegment_csv_path = csv_root / '032212_prep_210423_094825.resgment.csv'
    labelset = 'iabcdefghjk'

    predict_dst = 'results/hvc/predictions/gy6or6'

    pred_paths = article.hvc.predict(prep_csv_path,
                                     resegment_csv_path,
                                     clf_path,
                                     predict_dst,
                                     labelset)

    scores = article.hvc.score(prep_csv_path, pred_paths, None)

    #TODO: save scores


def get_segment_params(segment_params_ini, animal_id):
    config = configparser.ConfigParser()
    config.read(segment_params_ini)

    config_param_names = ['min_segment_dur', 'min_silent_interval', 'threshold']
    arg_param_names = ['min_syl_dur', 'min_silent_dur', 'threshold']

    seg_params = {}
    for config_param_name, arg_param_name in zip(config_param_names, arg_param_names):
        seg_params[arg_param_name] = float(config[config_param_name][animal_id])

    return seg_params


def main(results_root,
         animal_ids,
         segment_params_ini,
         csv_filename,
         to_annot):
    # import crowsetta here to avoid circular import
    import crowsetta

    results_root = Path(results_root)
    animal_id_roots = sorted([results_root / animal_id for animal_id in animal_ids])
    if not all([animal_id_root.exits() for animal_id_root in animal_id_roots]):
        doesnt_exist = [animal_id_root for animal_id_root in animal_id_roots if not animal_id_root.exits()]
        raise NotADirectoryError(
            f'directories for these animal IDs not found in results root:\n{doesnt_exist}'
        )

    # get segment parameters, fail early if they're not found
    all_segment_params = {}
    for animal_id in animal_ids:
        all_segment_params[animal_id] = get_segment_params(segment_params_ini,
                                                           animal_id)

    for animal_id_root in animal_id_roots:
        animal_id = animal_id_root.name
        segment_params = all_segment_params[animal_id]
        rerun_learncurve(animal_id_root,
                         segment_params)




PROJ_ROOT = pyprojroot.here()
BR_RESULTS_ROOT = PROJ_ROOT / 'results' / 'Bengalese_Finches' / 'learncurve'
SEGMENT_PARAMS_INI = PROJ_ROOT / 'data' / 'configs' / 'segment_parmas.ini'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root',
                        help=('root directory where subdirectories are individuals from dataset, '
                              'and each subdirectory contains results folders from a run '
                              'of `vak learncurve`'),
                        default=BR_RESULTS_ROOT)
    parser.add_argument('--animal_ids',
                        help=('string names of animal identifiers, should be directory names '
                              'within results_root'),
                        nargs='+')
    parser.add_argument('--segment_params_ini',
                        help=("path to .ini file with segmenting parameters "
                              "for audio files from each animal"),
                        default=SEGMENT_PARAMS_INI)
    parser.add_argument('--csv_filename',
                        help='filename of .csv that will be saved by this script in results_root',
                        default='error_across_birds_with_cleanup.csv')
    parser.add_argument('--to_annot',
                        help=("if this option is added, predictions will be converted to annotations "
                              "and then saved in an 'annotations' directory in 'results_root'. "
                              "(If the option is not added, then 'to_annot' defaults to False.)"),
                        action='store_true',
                        )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(results_root=args.results_root,
         animal_ids=args.animal_ids,
         segment_params_ini=args.segment_params_ini,
         csv_filename=args.csv_filename,
         to_annot=args.to_annot)
