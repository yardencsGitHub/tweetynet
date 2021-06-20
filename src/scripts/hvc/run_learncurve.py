"""re-runs a learncurve experiment, but using a support vector machine algorithm
with hand-engineered features extracted from pre-segmented syllables
"""
import argparse
import configparser
import json
from pathlib import Path

import hvc
import pandas as pd
import pyprojroot

import article.hvc


def run_hvc_expt(prep_csv_path,
                 segment_params,
                 labelset,
                 audio_format,
                 results_dst,
                 dummy_label='-',
                 spect_params_ref='tachibana',
                 logger=None,
                 ):
    """run a single ``hvc`` experiment

    Parameters
    ----------
    prep_csv_path : str, pathlib.Path
        to one of the .csv files saved by a run of
        ``vak learncurve`` -- where the training set
        is for one replicate of a specified duration,
        sampled randomly from the total training set
    segment_params : dict
        of parameters to use when re-segmenting
        the test set from ``prep_csv_path``
    labelset : set
        of labels, for segments in annotated vocalizations
    audio_format : str
        one of {'wav', 'cbin'}
    results_dst : pathlib.Path
        "results destination" where artifacts will be saved
    dummy_label : str
        dummy label given to segments when re-segmenting.
        Default is '-'. Used because labels cannot be empty.
    spect_params_ref : str
        Which parameters to use for spectrograms.
        Default is 'tachibana', that uses parameters
        from Tachibana et al. 2014

    Returns
    -------
    mean_segment_error_rate : float
        the mean segment error rate
    """
    from vak.logging import log_or_print

    # ---- make a bunch of paths right here at the top
    # (fail early if for some reason this doesn't work)
    annot_dst = results_dst / 'annot'
    csv_dst = results_dst / 'csv'
    features_dst = results_dst / 'features'
    resegment_features_dst = results_dst / 'features' / 'resegment'
    predict_dst = results_dst / 'predictions'
    scores_dst = results_dst / 'scores'
    clf_dst = results_dst / 'classifiers'

    for dst in (annot_dst,
                csv_dst,
                features_dst,
                resegment_features_dst,
                predict_dst,
                scores_dst,
                clf_dst):
        dst.mkdir(exist_ok=True, parents=True)

    # ---- 1. blindly resegment test set
    # (this is what would happen if dataset were not carefully hand annotated)
    log_or_print(
        f"(1) resegmenting test set",
        logger=logger,
        level="info",
    )
    resegment_csv_path = article.hvc.resegment.resegment(prep_csv_path,
                                                         segment_params,
                                                         dummy_label=dummy_label,
                                                         annot_dst=annot_dst,
                                                         csv_dst=csv_dst,
                                                         split='test')

    # ---- 2. extract features
    # from (correctly segmented) set
    # also need a `spect_maker` to call `hvc.audiofileIO.make_syls`
    log_or_print(
        f"(2) extracting features",
        logger=logger,
        level="info",
    )
    spect_params = hvc.parse.ref_spect_params.refs_dict[spect_params_ref]
    spect_params['window'] = 'Hann'
    spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)

    extract_csv_path = article.hvc.extract.extract(prep_csv_path,
                                                   labelset,
                                                   features_dst,
                                                   csv_dst,
                                                   spect_maker,
                                                   audio_format,
                                                   article.hvc.extract.FEATURE_LIST)

    # extract from re-segmented test set
    article.hvc.extract.extract(csv_path=resegment_csv_path,
                                labelset=dummy_label,
                                features_dst=resegment_features_dst,
                                csv_dst=csv_dst,
                                spect_maker=spect_maker,
                                audio_format=audio_format,
                                feature_list=article.hvc.extract.FEATURE_LIST)

    # ---- 3. train classifier
    log_or_print(
        f"(3) training classifiers",
        logger=logger,
        level="info",
    )
    clf = article.hvc.fit.fit(extract_csv_path,
                              learncurve_csv_path=prep_csv_path)

    clf_path = article.hvc.fit.save_clf(clf,
                                        extract_csv_path,
                                        clf_dst)

    # ---- 4. get predictions
    log_or_print(
        f"(4) getting predictions",
        logger=logger,
        level="info",
    )
    pred_paths = article.hvc.predict.predict(extract_csv_path,
                                             resegment_csv_path,
                                             clf_path,
                                             predict_dst,
                                             labelset)

    # ---- 5. compute scores on predictions
    log_or_print(
        f"(5) computing segment error rates",
        logger=logger,
        level="info",
    )
    scores = article.hvc.score.segment_error_rate(prep_csv_path,
                                                  pred_paths)
    scores = {source: scores_tuple._asdict() for source, scores_tuple in scores.items()}
    for source in scores:
        scores[source]['segment_error_rates'] = scores[source]['segment_error_rates'].tolist()
    scores_json_path = scores_dst / 'segment_error_rates.json'
    with scores_json_path.open('w') as fp:
        json.dump(scores, fp)

    return scores


def rerun_learncurve(previous_run_path,
                     segment_params,
                     results_dst,
                     logger=None):
    from vak import config  # import here to avoid circular imports
    from vak.core.learncurve import train_dur_csv_paths as _train_dur_csv_paths
    from vak.logging import log_or_print

    previous_run_path = Path(previous_run_path)
    toml_path = sorted(previous_run_path.glob('*.toml'))
    assert len(toml_path) == 1, f'found more than one .toml config file: {toml_path}'

    toml_path = toml_path[0]

    cfg = config.parse.from_toml_path(toml_path)

    # ---- get all the parameters from the config we need
    prep_csv_path = cfg.learncurve.csv_path
    labelset = cfg.prep.labelset
    audio_format = cfg.prep.audio_format

    # ---- make a "root" results directory for this animal id
    animal_id = previous_run_path.parent.name
    animal_id_results_dst = results_dst / animal_id
    animal_id_results_dst.mkdir()

    log_or_print(
        f"Loading previous training subsets from:\n{previous_run_path}",
        logger=logger,
        level="info",
    )

    records = []  # used for summary results DataFrame, append once every time through loop

    train_dur_csv_paths = _train_dur_csv_paths._dict_from_dir(previous_run_path)

    for train_dur, csv_paths in train_dur_csv_paths.items():
        # make train dur "root" within animal id root
        train_dur_results_dst = animal_id_results_dst / f'train_dur_{train_dur}'
        train_dur_results_dst.mkdir()
        for replicate_num, this_train_dur_this_replicate_csv_path in enumerate(
            csv_paths
        ):
            replicate_num += 1

            # finally make replicate "dst" within train dur "root"
            replicate_dst = train_dur_results_dst / f'replicate_{replicate_num}'
            replicate_dst.mkdir()

            this_train_dur_this_replicate_results_path = (
                this_train_dur_this_replicate_csv_path.parent
            )
            log_or_print(
                f"Training classifiers for training set duration of {train_dur},"
                f"replicate number {replicate_num}, "
                f"using dataset from .csv file:\n{this_train_dur_this_replicate_results_path}",
                logger=logger,
                level="info",
            )

            scores = run_hvc_expt(prep_csv_path=prep_csv_path,
                                  segment_params=segment_params,
                                  labelset=labelset,
                                  audio_format=audio_format,
                                  results_dst=replicate_dst,
                                  spect_params_ref='tachibana')

            for source in scores.keys():
                mean_seg_err_rate = scores[source]['mean_segment_error_rate']
                records.append(
                    {
                        'animal_id': animal_id,
                        'train_dur': train_dur,
                        'replicate': replicate_num,
                        'source': source,
                        'mean_segment_error_rate': mean_seg_err_rate,
                    }
                )
    df = pd.DataFrame.from_records(records)
    return df


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
         results_dst,
         csv_filename):
    """re-run learncurve experiment by training Support Vector Machine classifiers
    on engineered features extracted from segments, using the exact same splits
    that were used when training ``tweetynet`` models

    Parameters
    ----------
    results_root : str, pathlib.Path
        root directory with results from running
        ``vak learncurve`` experiments,
        where subdirectories are individuals from dataset,
        and each subdirectory contains results folders from a run
        of ``vak learncurve``
    animal_ids : list
        of str, animal identifiers.
        Should be names of subdirectories
        within ``results_root``.
        Specifies which of those subdirectories
        will be used.
    segment_params_ini : str
        path to 'segment_params.ini' file that specifies parameters
        to use when re-segmenting the test set, to benchmark
        performance on data segmented **without** the clean up
        that occurs during annotation by hand
    results_dst : str, pathlib.Path
        "results destination", where results from running this script
        should be saved
    """
    results_root = Path(results_root)
    animal_id_roots = sorted([results_root / animal_id for animal_id in animal_ids])
    if not all([animal_id_root.exists() for animal_id_root in animal_id_roots]):
        doesnt_exist = [animal_id_root for animal_id_root in animal_id_roots if not animal_id_root.exits()]
        raise NotADirectoryError(
            f'directories for these animal IDs not found in results root:\n{doesnt_exist}'
        )

    segment_params_ini = Path(segment_params_ini)
    if not segment_params_ini.exists():
        raise FileNotFoundError(f'segment_params_ini file not found: {segment_params_ini}')

    # get segment parameters, fail early if they're not found
    all_segment_params = {}
    for animal_id in animal_ids:
        all_segment_params[animal_id] = get_segment_params(segment_params_ini,
                                                           animal_id)

    dfs = []
    for animal_id_root in animal_id_roots:
        results_dirs = sorted(animal_id_root.glob('results_*'))
        most_recent_results = results_dirs[-1]

        print(
            f'running experiment for animal id "{animal_id_root.name}"\n'
            f'with results from directory: {most_recent_results.name}'
        )
        df = rerun_learncurve(previous_run_path=most_recent_results,
                              segment_params=all_segment_params[animal_id_root.name],
                              results_dst=results_dst)
        dfs.append(df)

    dfs = pd.concat(dfs)
    hvc_summary_csv_path = results_dst / csv_filename
    dfs.to_csv(hvc_summary_csv_path)


PROJ_ROOT = pyprojroot.here()
BR_RESULTS_ROOT = PROJ_ROOT / 'results' / 'Bengalese_Finches' / 'learncurve'
SEGMENT_PARAMS_INI = PROJ_ROOT / 'data' / 'configs' / 'segment_params.ini'
DEFAULT_RESULTS_DST = PROJ_ROOT / 'results' / 'Bengalese_Finches' / 'hvc'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root',
                        help=('root directory with results from running learncurve experiments, '
                              ' where subdirectories are individuals from dataset, '
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
                        default='segment_error_across_birds.hvc.csv')
    parser.add_argument('--results_dst',
                        help=('"results destination", directory where results from '
                              'running this script should be saved.'),
                        default=DEFAULT_RESULTS_DST)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(results_root=args.results_root,
         animal_ids=args.animal_ids,
         segment_params_ini=args.segment_params_ini,
         csv_filename=args.csv_filename,
         results_dst=args.results_dst)
