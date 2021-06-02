from pathlib import Path

import crowsetta
import evfuncs
import numpy as np
import pandas as pd
from tqdm import tqdm


def resegment(prep_csv,
              segment_params,
              annot_dst,
              csv_dst,
              split=None):
    prep_csv = Path(prep_csv)
    annot_dst = Path(annot_dst)
    csv_dst = Path(csv_dst)

    vak_df = pd.read_csv(prep_csv)
    if split is not None:
        vak_df = vak_df[vak_df.split == split]

    annot_path = annot_dst / (prep_csv.stem + '.resegment.annot.csv')
    annot_path = annot_path.resolve()

    audio_paths = vak_df.audio_path.values
    annots = []  # of generated files, will add to csv
    pbar = tqdm(audio_paths)
    n_audio = len(audio_paths)
    for ind, audio_path in enumerate(pbar):
        pbar.set_description(
            f'resegmenting audio file {ind + 1} of {n_audio}:{Path(audio_path).name}'
        )
        rawsong, samp_freq = evfuncs.load_cbin(audio_path)
        smooth = evfuncs.smooth_data(rawsong, samp_freq)
        onsets_s, offsets_s = evfuncs.segment_song(smooth, samp_freq, **segment_params)
        labels = np.array(list('-' * onsets_s.shape[0]))  # dummy labels
        seq = crowsetta.Sequence.from_keyword(onsets_s=onsets_s, offsets_s=offsets_s, labels=labels)
        annot = crowsetta.Annotation(annot_path=annot_path,
                                     audio_path=audio_path,
                                     seq=seq)
        annots.append(annot)

    scribe = crowsetta.Transcriber(format='csv')
    scribe.to_csv(annots, str(annot_path))
    annot_paths = [annot_path for _ in audio_paths]
    vak_df['annot_path'] = annot_paths
    vak_df['annot_format'] = 'csv'
    csv_path = csv_dst / (prep_csv.stem + '.resgment.csv')
    vak_df.to_csv(csv_path)
