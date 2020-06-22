from pathlib import Path
import json

import attr
import joblib
import numpy as np

import crowsetta
import vak
from vak.utils.data import reshape_data_for_batching

BIRDS = ['bl26lb16',
         'gy6or6',
         'or60yw70',
         'gr41rd51',
         ]

TRAIN_SET_DURATIONS = [60, 120, 480]

HERE = Path(__file__).parent
DATA_DIR = HERE.joinpath('../../data/BFSongRepository')

def main():
    for train_set_dur in TRAIN_SET_DURATIONS:
        for bird in BIRDS:
            all_predict_vds_paths = DATA_DIR.joinpath(f'{train_set_dur}s').joinpath(bird).joinpath('vds').glob('*predict.vds.json')
            for predict_vds_path in all_predict_vds_paths:
                print(f'resegmenting {predict_vds_path}')
                predict_vds = vak.Dataset.load(predict_vds_path)
                new_predict_vds_path = str(predict_vds_path).replace('predict.vds.json', 'predict.resegment.vds.json')
                if not Path(new_predict_vds_path).exists():
                    predict_vds = predict_vds.load_spects()

                    lbl_tb_predict = predict_vds.lbl_tb_list()
                    lbl_tb_predict_reseg = [vak.utils.labels.resegment(lbl_tb,min_dur_tb=2,majority_vote=True) for lbl_tb in lbl_tb_predict]

                    labels_reseg, onsets_reseg, offsets_reseg = [], [], []
                    for lbl_tb in lbl_tb_predict_reseg:
                        lbl, on, off = vak.utils.labels.lbl_tb2segments(lbl_tb,labelmap=predict_vds.labelmap,timebin_dur=predict_vds.voc_list[0].metaspect.timebin_dur)
                        labels_reseg.append(lbl)
                        onsets_reseg.append(on)
                        offsets_reseg.append(off)

                    new_annots = []
                    for lbl, on, off, voc in zip(labels_reseg, onsets_reseg, offsets_reseg, predict_vds.voc_list):
                        annot = crowsetta.Sequence.from_keyword(labels=lbl, onsets_s=on, offsets_s=off, file=voc.annot.file)
                        new_annots.append(annot)

                    new_voc_list = [attr.evolve(voc, annot=annot) for voc, annot in zip(predict_vds.voc_list, new_annots)]
                    predict_vds_reseg = attr.evolve(predict_vds, voc_list=new_voc_list)


                    print(f'saving resegmented Dataset in {new_predict_vds_path}')                
                    predict_vds_reseg = predict_vds_reseg.clear_spects()
                    predict_vds_reseg.save(new_predict_vds_path)
                else:
                    print(f'skipping {new_predict_vds_path}, already exists')

if __name__ == '__main__':
    main()

