"""module for converting Yarden's annotation.mat files into Crowsetta sequences"""
import numpy as np
from scipy.io import loadmat

from crowsetta.sequence import Sequence
from crowsetta.meta import Meta


def _cast_to_arr(val):
    """helper function that casts single elements to 1-d numpy arrays"""
    if type(val) == int or type(val) == float:
        # this happens when there's only one syllable in the file
        # with only one corresponding label
        return np.asarray([val])  # so make it a one-element list
    elif type(val) == np.ndarray:
        # this should happen whenever there's more than one label
        return val
    else:
        # something unexpected happened
        raise TypeError(f"Type {type(val)} not recognized.")


def yarden2seq(file,
               abspath=False,
               basename=False,
               round_times=True,
               decimals=3,
               fname_key='keys',
               annot_key='elements',
               onsets_key='segFileStartTimes',
               offsets_key='segFileEndTimes',
               labels_key='segType',
               samp_freq_key='fs'):
    """unpack annotation.mat file into list of Sequence objects

    Parameters
    ----------
    file : str
        path to .mat file of annotations, containing 'keys' and 'elements'
        where 'keys' are filenames of audio files and 'elements'
        contains additional annotation not found in .mat files
    abspath : bool
        if True, converts filename for each audio file into absolute path.
        Default is False.
    basename : bool
        if True, discard any information about path and just use file name.
        Default is False.
    round_times : bool
        if True, round onsets_s and offsets_s.
        Default is True.
    decimals : int
        number of decimals places to round floating point numbers to.
        Only meaningful if round_times is True.
        Default is 3, so that times are rounded to milliseconds.
    fname_key : str
        name of array in .mat file that lists filenames of .mat files
        containing spectrograms. Accessed by using the array name as a 
        key into a dictionary-like object, hence the name 'fname_key'.
        Default is 'keys'.
    annot_key : str
        name of array in .mat file that holds annotations for .mat files
        containing spectrograms. Default is 'elements'.
    onsets_key : str
        name of array in annotations that holds segment onset times in seconds.
        Defalt is 'segFileStartTimes'.
    offsets_key : str
        name of array in annotations that holds segment offset times in seconds.
        Defalt is 'segFileStartTimes'.
    labels_key : str
        name of array in annotations that holds label times in seconds.
        Defalt is 'segType'.
    samp_freq_key : str
        name of array in annotations that holds sample frequency of audio file.
        Defalt is 'fs'.

    Returns
    -------
    seq : list
        of Sequence objects

    Notes
    -----
    The abspath and basename parameters specify how file names for audio files are saved.
    These options are useful for working with multiple copies of files and for
    reproducibility. Default for both is False, in which case the filename is saved just
    as it is passed to this function.

    round_times and decimals arguments are provided to reduce differences across platforms
    due to floating point error, e.g. when loading .not.mat files and then sending them to
    a csv file, the result should be the same on Windows and Linux
    """
    if abspath and basename:
        raise ValueError('abspath and basename arguments cannot both be set to True, '
                         'unclear whether absolute path should be saved or if no path '
                         'information (just base filename) should be saved.')

    annot_mat = loadmat(file, squeeze_me=True)
    filenames = annot_mat[fname_key]
    annotations = annot_mat[annot_key]
    if len(filenames) != len(annotations):
        raise ValueError(f'list of filenames and list of annotations in {file} do not have the same length')

    seq_list = []
    # annotation structure loads as a Python dictionary with two keys
    # one maps to a list of filenames,
    # and the other to a Numpy array where each element is the annotation
    # coresponding to the filename at the same index in the list.
    # We can iterate over both by using the zip() function.
    for filename, annotation in zip(filenames, annotations):
        # below, .tolist() does not actually create a list,
        # instead gets ndarray out of a zero-length ndarray of dtype=object.
        # This is just weirdness that results from loading complicated data
        # structure in .mat file.
        seq_dict = {}
        seq_dict['onsets_s'] = annotation[onsets_key].tolist()
        seq_dict['offsets_s'] = annotation[offsets_key].tolist()
        seq_dict['labels'] = annotation[labels_key].tolist()
        seq_dict = dict((k, _cast_to_arr(seq_dict[k])) 
                        for k in ['onsets_s', 'offsets_s', 'labels'])
        # we want to wait to add file to seq dict until *after* casting all values in dict to numpy arrays
        seq_dict['file'] = filename
        samp_freq = annotation[samp_freq_key].tolist()
        seq_dict['onsets_Hz'] = np.round(seq_dict['onsets_s'] * samp_freq).astype(int)
        seq_dict['offsets_Hz'] = np.round(seq_dict['offsets_s'] * samp_freq).astype(int)

        # do this *after* converting onsets_s and offsets_s to onsets_Hz and offsets_Hz
        # probably doesn't matter but why introduce more noise?
        if round_times:
            seq_dict['onsets_Hz'] = np.around(seq_dict['onsets_Hz'], decimals=decimals)
            seq_dict['offsets_Hz'] = np.around(seq_dict['offsets_Hz'], decimals=decimals)

        seq = Sequence.from_dict(seq_dict)
        seq_list.append(seq)

    return seq_list


meta = Meta(
    name='yarden',
    ext='.mat',
    to_seq=yarden2seq
)
