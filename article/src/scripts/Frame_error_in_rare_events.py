#!/usr/bin/env python
# coding: utf-8

# # Frame error in rare events
# This notebook calculates frame error exclusively in pre-defined time epochs of rare events
# \
# Yarden, June 2021

# In[1]:


# imports
from argparse import ArgumentParser
import configparser  # used to load 'min_segment_dur.ini'

from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyprojroot
import torch
from tqdm import tqdm

from vak import config, io, models, transforms
from vak.datasets.vocal_dataset import VocalDataset
import vak.device
import vak.files
from vak.labeled_timebins import lbl_tb2segments, majority_vote_transform, lbl_tb_segment_inds_list,     remove_short_segments
from vak.core.learncurve import train_dur_csv_paths as _train_dur_csv_paths
from vak.logging import log_or_print
from vak.labeled_timebins import (
    lbl_tb2segments,
    majority_vote_transform,
    lbl_tb_segment_inds_list,
    remove_short_segments
)
import copy
from collections import Counter
from crowsetta import Transcriber
from pathlib import Path


# In[47]:


# Data folders and parameters
min_segment_dur_ini = 'D:\\Users\\yarde\\github\\tweetynet\\data\\configs\\min_segment_dur.ini  '
config = configparser.ConfigParser()
config.optionxform = lambda option: option  # monkeypatch optionxform so it doesn't lowercase animal IDs
config.read(Path(min_segment_dur_ini).expanduser().resolve())
min_segment_durs = {k: float(v) for k, v in config['min_segment_dur'].items()}

Root_learning_curve = Path('D:\\Users\\yarde\\vak_project\\BF\\learncurve')
Root_behavior = Path('D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository')
Root_hidden_16 = Path('D:\\Users\\yarde\\vak_project\\BF\\hidden_size\\hidden_size_16')
Root_hidden_64 = Path('D:\\Users\\yarde\\vak_project\\BF\\hidden_size\\hidden_size_64')
birds = ['bl26lb16','gr41rd51','gy6or6','or60yw70']

# output folder
output_folder = Path('D:\\Users\\yarde\\github\\tweetynet\\results\\Bengalese_Finches\\rare_events')

# general parameters
min_p_ratio = 0.001
min_rare_count = 10
min_count = 50
max_p_ratio = 0.25


# In[3]:


# functions to convert .not.mat annotations for a single .csv
def convert_notmats_to_csv(notmat_folder,csv_filename):
    scribe = Transcriber(format='notmat')
    annotpaths = [str(x) for x in Path(notmat_folder).glob('*.not.mat')]
    scribe.to_csv(annotpaths,csv_filename)


# In[4]:


# creaate csv annotations for BFSongRepository
BF1_notmat_folder = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\bl26lb16\\041912'
BF1_csv_filename = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\bl26lb16\\bl26lb16_annotation.csv'
convert_notmats_to_csv(BF1_notmat_folder,BF1_csv_filename)
BF2_notmat_folder = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\gr41rd51\\06-21-12'
BF2_csv_filename = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\gr41rd51\\gr41rd51_annotation.csv'
convert_notmats_to_csv(BF2_notmat_folder,BF2_csv_filename)
BF3_notmat_folder = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\gy6or6\\032212'
BF3_csv_filename = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\gy6or6\\gy6or6_annotation.csv'
convert_notmats_to_csv(BF3_notmat_folder,BF3_csv_filename)
BF4_notmat_folder = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\or60yw70\\09-27-28-12'
BF4_csv_filename = 'D:\\Users\\yarde\\vak_project\\BF\\BFSongRepository\\or60yw70\\or60yw70_annotation.csv'
convert_notmats_to_csv(BF4_notmat_folder,BF4_csv_filename)


# In[5]:


# Bird names and Root folders for experiments:
Root_learning_curve = Path('D:\\Users\\yarde\\vak_project\\BF\\learncurve')
Root_hidden_16 = Path('D:\\Users\\yarde\\vak_project\\BF\\hidden_size\\hidden_size_16')
Root_hidden_64 = Path('D:\\Users\\yarde\\vak_project\\BF\\hidden_size\\hidden_size_64')
birds = ['bl26lb16','gr41rd51','gy6or6','or60yw70']
annotations = {'bl26lb16':BF1_csv_filename,'gr41rd51':BF2_csv_filename,'gy6or6':BF3_csv_filename,'or60yw70':BF4_csv_filename}


# In[6]:


# function to locate rare events
def locate_rare_events(path_annot_csv,labelmap,degree=3):
    if 'unlabeled' in labelmap.keys():
        unl_shift = 1
    else:
        unl_shift = 0
    inverse_labelmap = dict((v, k) for k, v in labelmap.items())
    annot_df = pd.read_csv(path_annot_csv)
    filenames = np.unique(annot_df.audio_path)
    labels = "".join([l for l in labelmap.keys() if l != 'unlabeled'])
    nsyls = len(labels)
    # create ngram matrix
    if degree==3:
        transmat = np.zeros((nsyls,nsyls,nsyls))
    else:
        transmat = np.zeros((nsyls,nsyls))
    for filename in filenames:
        label_idx_seq = np.array([labelmap[x]-unl_shift for x in annot_df[annot_df.audio_path==filename].label if x in labelmap.keys()])
    
        if degree==3:
            for i in range(len(label_idx_seq)-2):
                a=label_idx_seq[i]; b=label_idx_seq[i+1]; c=label_idx_seq[i+2]
                transmat[a,b,c] +=1
        else:
            for i in range(len(label_idx_seq)-1):
                a=label_idx_seq[i]; b=label_idx_seq[i+1];
                transmat[a,b] +=1
        
    # find forking transition points
    if degree==3:
        syl1 = []
        syl2 = []
        outsyls = []
        for a in range(unl_shift,nsyls):
            for b in range(unl_shift,nsyls):
                if sum(np.squeeze(transmat[a-unl_shift,b-unl_shift,:]) > 0) > 1:
                    syl1.append(inverse_labelmap[a])
                    syl2.append(inverse_labelmap[b])
                    outsyls.append(np.squeeze(transmat[a-unl_shift,b-unl_shift,:]))
        rare_events_df = pd.DataFrame({'a':syl1,'b':syl2,'trans_outcome':outsyls})
    else:
        syl1 = []
        outsyls = []
        for a in range(unl_shift,nsyls):
            if sum(np.squeeze(transmat[a-unl_shift,:]) > 0) > 1:
                syl1.append(inverse_labelmap[a]);
                outsyls.append(np.squeeze(transmat[a0unl_shift,:]))
        rare_events_df = pd.DataFrame({'a':syl1,'trans_outcome':outsyls})    
    return rare_events_df
    


# In[7]:


# Curate 2nd and 3rd order rare events


# In[8]:


def load_network_results(path_to_config=None,
                        spect_scaler_path = None,
                        csv_path=None,
                        labelmap_path=None,
                        checkpoint_path=None,
                        window_size = 370,
                        hidden_size = None,
                        min_segment_dur = 0.01,
                        num_workers = 12,
                        device='cuda',
                        spect_key='s',
                        timebins_key='t',
                        freq_key = 'f',
                        test_all_files=False):
    '''
    This function loads a model from an EVAL config file or from specified parameters, loads a model, and returns its outputs 
    for a specified test set.
    
    Setting 'test_all_files=True' will create a copy of the list in csv_path where all files are in the test set.
    '''
    if path_to_config:
        # ---- get all the parameters from the config we need
        cfg = config.parse.from_toml_path(path_to_config)
        if cfg.eval: 
            model_config_map = config.models.map_from_path(path_to_config, cfg.eval.models)
            csv_path = cfg.eval.csv_path
            labelmap_path = cfg.eval.labelmap_path
            checkpoint_path = cfg.eval.checkpoint_path
            window_size = cfg.dataloader.window_size
            num_workers = cfg.eval.num_workers
            if spect_scaler_path:
                spect_scaler_path = cfg.eval.spect_scaler_path
        else:
            print('config file must hold parameters in an [EVAL] section')
            return None
    else:
        if hidden_size:
            model_config_map = {'TweetyNet': {'loss': {}, 'metrics': {}, 'network': {'hidden_size':hidden_size}, 'optimizer': {'lr': 0.001}}}
        else:
            model_config_map = {'TweetyNet': {'loss': {}, 'metrics': {}, 'network': {}, 'optimizer': {'lr': 0.001}}}
        
    with labelmap_path.open('r') as f:
        labelmap = json.load(f)
    if spect_scaler_path:
        spect_standardizer = joblib.load(spect_scaler_path)
    else:
        spect_standardizer = None
    # prepare evaluation data
    csv_df = pd.read_csv(csv_path)
    if test_all_files==True: # allow creating a new csv 'csv_path_test.csv' where all entries are 'test'
        csv_df['split'] = 'test'
        csv_df.to_csv(csv_path.parent.joinpath(csv_path.stem + '_test.csv'))
        csv_path = csv_path.parent.joinpath(csv_path.stem + '_test.csv')
    csv_df = csv_df[csv_df.split == 'test']
    
    item_transform = transforms.get_defaults('eval',
                                                 spect_standardizer=spect_standardizer,
                                                 window_size=window_size,
                                                 return_padding_mask=True,
                                                 )

    eval_dataset = VocalDataset.from_csv(csv_path=csv_path,
                                         split='test',
                                         labelmap=labelmap,
                                         spect_key=spect_key,
                                         timebins_key=timebins_key,
                                         item_transform=item_transform,
                                         )

    eval_data = torch.utils.data.DataLoader(dataset=eval_dataset,
                                            shuffle=False,
                                            # batch size 1 because each spectrogram reshaped into a batch of windows
                                            batch_size=1,
                                            num_workers=num_workers)
    input_shape = eval_dataset.shape
    # if dataset returns spectrogram reshaped into windows,
    # throw out the window dimension; just want to tell network (channels, height, width) shape
    if len(input_shape) == 4:
        input_shape = input_shape[1:]

    models_map = models.from_model_config_map(
        model_config_map,
        num_classes=len(labelmap),
        input_shape=input_shape
    )
    model_name = 'TweetyNet'
    model = models_map['TweetyNet']
    model.load(checkpoint_path)
    #metrics = model.metrics  # metric name -> callable map we use below in loop
    if device is None:
        device = vak.device.get_default_device()
    pred_dict = model.predict(pred_data=eval_data,
                              device=device)
    
    
    annotation_dfs = [pd.DataFrame(eval_dataset.annots[file_number].seq.as_dict()) for file_number in range(len(csv_df))]
   
    return csv_df, annotation_dfs, pred_dict, labelmap

def seq_in_seq(long_seq,target_seq):
    #import pdb
    #pdb.set_trace()
    onsets = []
    offsets = []
    for onset in np.arange(0,len(long_seq)-len(target_seq)):
        if list(long_seq)[onset:onset+len(target_seq)] == list(target_seq):
            onsets.append(onset)
            offsets.append(onset+len(target_seq))
    return onsets,offsets


# In[9]:


# Function to summarize experiments


# In[10]:


def create_results(annot_path,rare_trans_df,labelmap,csv_df,pred_dict,degree=3):
    if 'unlabeled' in labelmap.keys():
        unl_shift = 1
    else:
        unl_shift = 0
    labels = "".join([l for l in labelmap.keys() if l != 'unlabeled'])
    df_annot = pd.read_csv(annot_path)
    inverse_labelmap = dict((v, k) for k, v in labelmap.items())
    nsyls = len(labels)
    sequences = []
    ratios = []
    totals = []
    for n in range(len(rare_trans_df)):
        stem = np.array(rare_trans_df.loc[n,['a','b']])
        trans_n = np.array(rare_trans_df.loc[n,'trans_outcome'])
        tot_num_event = sum(trans_n)
        trans_p = trans_n/tot_num_event
        max_p = max(trans_p)
        rec_trans_p = trans_p/max_p
        if tot_num_event >= min_count:
            for i in range(len(rec_trans_p)):
                if ((rec_trans_p[i] <= max_p_ratio) & (trans_n[i] > min_rare_count) & (trans_p[i] > min_p_ratio)):
                    print('Transition added: ',stem,'to',inverse_labelmap[i+unl_shift])
                    ratios.append(rec_trans_p[i])
                    sequences.append(np.append(stem,inverse_labelmap[i+unl_shift]))
                    totals.append(trans_n)

    names = [Path(x).name for x in csv_df['audio_path']]
    spect_names = [x for x in csv_df['spect_path']]

    idxs = []
    times_on=[]
    times_off=[]
    seq_idxs = []
    for ind,name in enumerate(names):
        temp_df = copy.deepcopy(df_annot[[Path(x).name == name for x in df_annot.audio_path]])
        for seq_id,seq in enumerate(sequences):
            onsets,offsets = seq_in_seq(np.array(temp_df.label),seq)
            if len(onsets)>0:
                idxs.append(ind)
                times_on.append([temp_df['onset_s'].iloc[x-1] for x in offsets])
                times_off.append([temp_df['offset_s'].iloc[x-1] for x in offsets])
                seq_idxs.append(seq_id)
                #print('seq:',seq,'in',name,':',times_on[-1])


    # now collect frame error rate

    argmax_labels = []
    argmax_labels_maj = []
    cnt=0
    for idx,seq_idx,t_ons,t_offs in zip(idxs,seq_idxs,times_on,times_off):
        #print(cnt)
        #spect = vak.files.spect.load(str(spect_names[idx]))['s']
        model_output = pred_dict[str(spect_names[idx])]
        model_output = np.squeeze(model_output.cpu().numpy())
        model_output = np.transpose(model_output,(0,2,1))
        m_shape = np.shape(model_output)
        model_output = model_output.reshape(m_shape[0]*m_shape[1],m_shape[2])
        t_vec = vak.files.spect.load(str(spect_names[idx]))['t']#[0] #remember to remove [0]
        f_vec = vak.files.spect.load(str(spect_names[idx]))['f']#[0]   
        model_output = model_output[:len(t_vec)]
        #import pdb
        #pdb.set_trace()
        #model_output_argmax = np.array([int(inverse_labelmap[(x)]) if x>0 else 0 for x in np.argmax(model_output,axis=1)])
        model_output_argmax = np.argmax(model_output,axis=1)
        tmp = [model_output_argmax[(t_vec >= t_on) & (t_vec <= t_off)] for t_on,t_off in zip(t_ons,t_offs)]
        argmax_labels.append(np.concatenate(tmp))
        tmp_maj = []
        for t in tmp:
            tmp_maj.append(list(Counter(t).most_common(1)[0])[0]*np.ones_like(t))
        argmax_labels_maj.append(np.concatenate(tmp_maj))
        
        
    errs_maj = []
    errs=[]
    ns = []
    for i_seq,seq in enumerate(sequences):
        ratio = ratios[i_seq]
        label = seq[-1]
        tmp_argmax_seq_labels = [[inverse_labelmap[y] for y in argmax_labels[x]] for x in np.where(np.array(seq_idxs) == i_seq)[0]]
        if len(tmp_argmax_seq_labels) > 0:
            argmax_seq_labels = np.concatenate(tmp_argmax_seq_labels)
            print('seq:',seq,'ratio:',ratio,'err',1-np.mean(argmax_seq_labels == label),'n',totals[i_seq])
            errs.append(1-np.mean(argmax_seq_labels == label))
        else:
            print('seq:',seq,'does not appear in the test set')
            errs.append(None)
        tmp_argmax_seq_labels = [[inverse_labelmap[y] for y in argmax_labels_maj[x]] for x in np.where(np.array(seq_idxs) == i_seq)[0]]
        if len(tmp_argmax_seq_labels) > 0:
            argmax_seq_labels = np.concatenate(tmp_argmax_seq_labels)
            errs_maj.append(1-np.mean(argmax_seq_labels == label))
            ns.append(len(argmax_seq_labels))
        else:
            #print('seq:',seq,'does not appear in the test set')
            errs_maj.append(None)
            ns.append(None)
    outdict={'sequences':sequences, 
             'ratios':ratios, 
             'totals':totals, 
             'idxs':idxs, 
             'times_on':times_on, 
             'times_off':times_off, 
             'seq_idxs':seq_idxs,  
             'argmax_labels':argmax_labels, 
             'argmax_labels_maj':argmax_labels_maj, 
             'errs_maj':errs_maj, 
             'errs':errs, 
             'ns':ns}    
    return outdict


# In[11]:


# summarize results with hidden_size=16
window_size = 176
hidden_size = 16
training_dur_summary = []
ratio_summary = []
errs_summary = []
errs_maj_summary = []
errs_se_summary = []
errs_se_maj_summary = []

for bird in birds:
    min_segment_dur = min_segment_durs[bird]
    result_folder = [d for d in Root_hidden_16.joinpath(bird).iterdir()][0]
    train_dur_folders = [d for d in result_folder.iterdir() if d.is_dir()]
    for curr_train_folder in train_dur_folders:
        replicate_folders = [d for d in curr_train_folder.iterdir() if d.is_dir()]
        ods = []
        for curr_replicate_folder in replicate_folders:
            path_labelmap = curr_replicate_folder.joinpath('labelmap.json')
            if curr_replicate_folder.joinpath('StandardizeSpect').exists():
                spect_scaler_path = curr_replicate_folder.joinpath('StandardizeSpect')
            else:
                spect_scaler_path = None

            checkpoint_path = curr_replicate_folder.joinpath('TweetyNet','checkpoints','max-val-acc-checkpoint.pt')
            csv_path = [f for f in Root_learning_curve.joinpath(bird).glob('*.csv')][0]
            csv_df, annotation_dfs, pred_dict, labelmap = load_network_results(path_to_config=None,
                                                                             spect_scaler_path = spect_scaler_path,
                                                                             csv_path=csv_path,
                                                                             labelmap_path=path_labelmap,
                                                                             checkpoint_path=checkpoint_path,
                                                                             window_size = window_size,
                                                                             hidden_size = hidden_size,
                                                                             min_segment_dur = min_segment_dur,
                                                                             num_workers = 4,
                                                                             device='cuda',
                                                                             spect_key='s',
                                                                             timebins_key='t',
                                                                             freq_key = 'f',
                                                                             test_all_files=False)
            rare_trans_df = locate_rare_events(annotations[bird],labelmap,degree=3)
            od = create_results(annotations[bird],rare_trans_df,labelmap,csv_df,pred_dict,degree=3)
            ods.append(od)
            print('Done',curr_replicate_folder)
        #locate_rare_events(BF1_csv_filename,'output_folder',labelmap,degree=3)
        mn=np.nanmean(np.array([x['errs'] for x in ods]).astype(float),axis=0)
        sd=np.nanstd(np.array([x['errs'] for x in ods]).astype(float),axis=0)
        mn_maj=np.nanmean(np.array([x['errs_maj'] for x in ods]).astype(float),axis=0)
        sd_maj=np.nanstd(np.array([x['errs_maj'] for x in ods]).astype(float),axis=0)
        
        errs_summary.append(mn)
        errs_maj_summary.append(mn_maj)
        errs_se_summary.append(sd/np.sqrt(10))
        errs_se_maj_summary.append(sd_maj/np.sqrt(10))
        
        ratio_summary.append(ods[0]['ratios'])
        
        training_dur_summary.append(curr_train_folder.parts[-1].split('_')[-1])
#%%


# In[108]:





# In[46]:


#(training_dur_summary[:7])
#(ratio_summary[:7])
#errs_summary 
#errs_maj_summary 
#errs_se_summary 
#errs_se_maj_summary 
data = dict((x,np.concatenate(np.array(errs_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj = dict((x,np.concatenate(np.array(errs_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))

#se
data_se = dict((x,np.concatenate(np.array(errs_se_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj_se = dict((x,np.concatenate(np.array(errs_se_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))


all_ratios = dict((x,np.concatenate(np.array(ratio_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
durs = np.array([int(s[:-1]) for s in data.keys()])
durkeys = np.array([s for s in data.keys()])


# In[48]:


# convert to data frames and save .csv source data
# One bird did not have the 600s datapoint so adding 'nan's 

all_ratios['600s'] = np.concatenate([all_ratios['600s'],[np.nan,np.nan]])
all_ratios_df = pd.DataFrame(all_ratios)

data['600s'] = np.concatenate([data['600s'],[np.nan,np.nan]])
data_df = pd.DataFrame(data)

data_maj['600s'] = np.concatenate([data_maj['600s'],[np.nan,np.nan]])
data_maj_df = pd.DataFrame(data_maj)

data_se['600s'] = np.concatenate([data_se['600s'],[np.nan,np.nan]])
data_se_df = pd.DataFrame(data_se)

data_maj_se['600s'] = np.concatenate([data_maj_se['600s'],[np.nan,np.nan]])
data_maj_se_df = pd.DataFrame(data_maj_se)

all_ratios_df.to_csv(output_folder.joinpath('all_ratios_hidden_16.csv'))
data_df.to_csv(output_folder.joinpath('data_hidden_16.csv'))
data_maj_df.to_csv(output_folder.joinpath('data_maj_hidden_16.csv'))
data_se_df.to_csv(output_folder.joinpath('data_se_hidden_16.csv'))
data_maj_se_df.to_csv(output_folder.joinpath('data_maj_se_hidden_16.csv'))


# In[13]:


#(training_dur_summary[:7])
#(ratio_summary[:7])
#errs_summary 
#errs_maj_summary 
#errs_se_summary 
#errs_se_maj_summary 
result_folder = 'D:\\Users\\yarde\\vak_project\\BF\\hidden_size'
data = dict((x,np.concatenate(np.array(errs_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj = dict((x,np.concatenate(np.array(errs_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
all_ratios = dict((x,np.concatenate(np.array(ratio_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
durs = np.array([int(s[:-1]) for s in data.keys()])
durkeys = np.array([s for s in data.keys()])
sortind = np.argsort(durs)
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
plt.figure(figsize=(25,4))
for i,e in enumerate(durkeys[sortind]):
    ax = plt.subplot(1,len(durs),i+1)
    ax.scatter(all_ratios[e],data[e])
    ax.scatter(all_ratios[e],data_maj[e])
    x = [all_ratios[e][pos] for pos,d in enumerate(data[e]) if not np.isnan(d)]
    y = [data[e][pos] for pos,d in enumerate(data[e]) if not np.isnan(d)]
    r,p = pearsonr(x,y)
    ax.set_title(e + ': r={0:1.2f},p={1:1.2f}'.format(r,p))
    ax.set_ylim([0,1])
    if i>0: 
        ax.set_yticks([])
    else:
        ax.set_ylabel('Frame error rate')
        ax.set_xlabel('Ratio of Rare/Frequent sequence')
        ax.legend(['raw','maj_vote'])
plt.suptitle('Error rates in rare sequences - Summary plot for BFSongRepository data with num_hidden = 16',fontsize=16)  
plt.tight_layout()
filename = 'rare_events_test_hidden_16'
plt.savefig(result_folder + '\\' + filename + '.png')
plt.savefig(result_folder + '\\' + filename + '.pdf')
plt.show()


# In[169]:





# ## Conclusion - for hidden_num=16:
# #### 1. We examine the frame error in syllables 'X' in the sequences 'a-b-X' and look for relation to the rareity of 'a-b-X' compared to the most frequent alternative 'a-b-Y'
# #### 2. High error rates in rare sequences are more likely to occur in very rare events but there is no significant correlation (pearsons r,p - high r values resulting from a few outliers).
# #### 3. The trend depends on how well the model is trained (the duration of the training set)
# #### 4. Using the majority vote cleanup almost always reduce the error rates.
# #### 5. The minimum of the correlation coefficient may be another sign of the optimal training set duration (for future developments)
# 
# 

# In[49]:


# summarize results with hidden_size=64
window_size = 176
hidden_size = 64
training_dur_summary = []
ratio_summary = []
errs_summary = []
errs_maj_summary = []
errs_se_summary = []
errs_se_maj_summary = []

for bird in birds:
    min_segment_dur = min_segment_durs[bird]
    result_folder = [d for d in Root_hidden_64.joinpath(bird).iterdir()][0]
    train_dur_folders = [d for d in result_folder.iterdir() if d.is_dir()]
    for curr_train_folder in train_dur_folders:
        replicate_folders = [d for d in curr_train_folder.iterdir() if d.is_dir()]
        ods = []
        for curr_replicate_folder in replicate_folders:
            path_labelmap = curr_replicate_folder.joinpath('labelmap.json')
            if curr_replicate_folder.joinpath('StandardizeSpect').exists():
                spect_scaler_path = curr_replicate_folder.joinpath('StandardizeSpect')
            else:
                spect_scaler_path = None

            checkpoint_path = curr_replicate_folder.joinpath('TweetyNet','checkpoints','max-val-acc-checkpoint.pt')
            csv_path = [f for f in Root_learning_curve.joinpath(bird).glob('*.csv')][0]
            csv_df, annotation_dfs, pred_dict, labelmap = load_network_results(path_to_config=None,
                                                                             spect_scaler_path = spect_scaler_path,
                                                                             csv_path=csv_path,
                                                                             labelmap_path=path_labelmap,
                                                                             checkpoint_path=checkpoint_path,
                                                                             window_size = window_size,
                                                                             hidden_size = hidden_size,
                                                                             min_segment_dur = min_segment_dur,
                                                                             num_workers = 4,
                                                                             device='cuda',
                                                                             spect_key='s',
                                                                             timebins_key='t',
                                                                             freq_key = 'f',
                                                                             test_all_files=False)
            rare_trans_df = locate_rare_events(annotations[bird],labelmap,degree=3)
            od = create_results(annotations[bird],rare_trans_df,labelmap,csv_df,pred_dict,degree=3)
            ods.append(od)
            print('Done',curr_replicate_folder)
        #locate_rare_events(BF1_csv_filename,'output_folder',labelmap,degree=3)
        mn=np.nanmean(np.array([x['errs'] for x in ods]).astype(float),axis=0)
        sd=np.nanstd(np.array([x['errs'] for x in ods]).astype(float),axis=0)
        mn_maj=np.nanmean(np.array([x['errs_maj'] for x in ods]).astype(float),axis=0)
        sd_maj=np.nanstd(np.array([x['errs_maj'] for x in ods]).astype(float),axis=0)
        
        errs_summary.append(mn)
        errs_maj_summary.append(mn_maj)
        errs_se_summary.append(sd/np.sqrt(10))
        errs_se_maj_summary.append(sd_maj/np.sqrt(10))
        
        ratio_summary.append(ods[0]['ratios'])
        
        training_dur_summary.append(curr_train_folder.parts[-1].split('_')[-1])


# In[50]:


# Summarize for hidden_size=64
result_folder = 'D:\\Users\\yarde\\vak_project\\BF\\hidden_size'
#(training_dur_summary[:7])
#(ratio_summary[:7])
#errs_summary 
#errs_maj_summary 
#errs_se_summary 
#errs_se_maj_summary 
data = dict((x,np.concatenate(np.array(errs_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj = dict((x,np.concatenate(np.array(errs_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
all_ratios = dict((x,np.concatenate(np.array(ratio_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
durs = np.array([int(s[:-1]) for s in data.keys()])
durkeys = np.array([s for s in data.keys()])
sortind = np.argsort(durs)
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
plt.figure(figsize=(25,4))
for i,e in enumerate(durkeys[sortind]):
    ax = plt.subplot(1,len(durs),i+1)
    ax.scatter(all_ratios[e],data[e])
    ax.scatter(all_ratios[e],data_maj[e])
    x = [all_ratios[e][pos] for pos,d in enumerate(data[e]) if not np.isnan(d)]
    y = [data[e][pos] for pos,d in enumerate(data[e]) if not np.isnan(d)]
    r,p = pearsonr(x,y)
    ax.set_title(e + ': r={0:1.2f},p={1:1.2f}'.format(r,p))
    ax.set_ylim([0,1])
    if i>0: 
        ax.set_yticks([])
    else:
        ax.set_ylabel('Frame error rate')
        ax.set_xlabel('Ratio of Rare/Frequent sequence')
        ax.legend(['raw','maj_vote'])
plt.suptitle('Error rates in rare sequences - Summary plot for BFSongRepository data with num_hidden = 64',fontsize=16)  
plt.tight_layout()
filename = 'rare_events_test_hidden_64'
plt.savefig(result_folder + '\\' + filename + '.png')
plt.savefig(result_folder + '\\' + filename + '.pdf')
plt.show()


# ## Conclusion - for hidden_num=64:
# #### 1. Compared to the hidden_size=16 case, error rates are smaller
# #### 2. The trend of more errors in rare events is smaller, almost not significant - the pearson 'r' values is determined by one outlier
# #### 3. Using the majority vote cleanup almost always reduce the error rates.
# #### 5. The larger hidden_size allows convergence to better results with shorter training duration.
# 

# In[51]:


data = dict((x,np.concatenate(np.array(errs_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj = dict((x,np.concatenate(np.array(errs_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))

#se
data_se = dict((x,np.concatenate(np.array(errs_se_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
data_maj_se = dict((x,np.concatenate(np.array(errs_se_maj_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))

all_ratios = dict((x,np.concatenate(np.array(ratio_summary,dtype=object)[np.where(np.array(training_dur_summary) == x)])) for x in np.unique(training_dur_summary))
durs = np.array([int(s[:-1]) for s in data.keys()])
durkeys = np.array([s for s in data.keys()])

# convert to data frames and save .csv source data
# One bird did not have the 600s datapoint so adding 'nan's 

all_ratios['600s'] = np.concatenate([all_ratios['600s'],[np.nan,np.nan]])
all_ratios_df = pd.DataFrame(all_ratios)

data['600s'] = np.concatenate([data['600s'],[np.nan,np.nan]])
data_df = pd.DataFrame(data)

data_maj['600s'] = np.concatenate([data_maj['600s'],[np.nan,np.nan]])
data_maj_df = pd.DataFrame(data_maj)

data_se['600s'] = np.concatenate([data_se['600s'],[np.nan,np.nan]])
data_se_df = pd.DataFrame(data_se)

data_maj_se['600s'] = np.concatenate([data_maj_se['600s'],[np.nan,np.nan]])
data_maj_se_df = pd.DataFrame(data_maj_se)

all_ratios_df.to_csv(output_folder.joinpath('all_ratios_hidden_64.csv'))
data_df.to_csv(output_folder.joinpath('data_hidden_64.csv'))
data_maj_df.to_csv(output_folder.joinpath('data_maj_hidden_64.csv'))
data_se_df.to_csv(output_folder.joinpath('data_se_hidden_64.csv'))
data_maj_se_df.to_csv(output_folder.joinpath('data_maj_se_hidden_64.csv'))


# In[ ]:




