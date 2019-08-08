#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pathlib import Path

import article


# In[2]:


config_dir = Path('../../../src/configs/')
config_files = config_dir.glob('*BirdsongRecognition*ini')
config_files = sorted([config_file for config_file in config_files])

data_dir = Path('../../../data/')
test_dirs = data_dir / 'BirdsongRecognition'
test_dirs = test_dirs.glob('Bird*/')
test_dirs = sorted([test_dir for test_dir in test_dirs])

csv_fname = str(Path('../../../results/BirdsongRecognition_test.csv'))


# In[3]:


df = article.util.make_df(config_files, test_dirs, 
                          net_name='TweetyNet', csv_fname=csv_fname, train_set_durs=[60, 120, 480])
agg_df = article.util.agg_df(df, [60, 120, 480])


# In[18]:


ax_frame_err = article.plot.frame_error_rate_test_mean(agg_df, save_as='../figures/fig2-frame-error.png')


# In[17]:


ax_syl_err = article.plot.syllable_error_rate_test_mean(agg_df, save_as='../figures/fig2-frame-error.png')


# In[ ]:




