[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2667812.svg)](https://doi.org/10.5281/zenodo.2667812)
[![PyPI version](https://badge.fury.io/py/tweetynet.svg)](https://badge.fury.io/py/tweetynet)

# TweetyNet
<p align="center"><img src="./doc/tweetynet.gif" alt="tweetynet image" width=400></p>

## A hybrid convolutional-recurrent neural network that segments and labels birdsong and other vocalizations.

![sample annotation](doc/sample_phrase_annotation.png)
Canary song segmented into phrases

## Installation
To install, run the following command at the command line:  
`pip install tweetynet`

Before you install, you'll want to set up a virtual environment
(for an explanation of why, see
https://www.geeksforgeeks.org/python-virtual-environment/).
Creating a virtual environment is not as hard as it might sound;
here's a primer: https://realpython.com/python-virtual-environments-a-primer/  
For many scientific packages that depend on libraries written in  
languages besides Python, you may find it easier to use 
a platform dedicated to managing those dependencies, such as
[Anaconda](https://www.anaconda.com/download) (which is free).
You can use the `conda` command-line tool developed by Anaconda  
to create environments and install the scientific libraries that this package 
depends on. In addition, using `conda` to install the dependencies may give you some performance gains 
(see https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/).  
Here's how you'd set up a `conda` environment:  
`/home/you/code/ $ conda create -n tweetyenv python=3.6 numpy scipy joblib tensorflow-gpu ipython jupyter`    
`/home/you/code/ $ source activate tweetyenv`  
(You don't have to `source` on Windows: `> activate tweetyenv`)  

You can then use `pip` inside a `conda` environment:  
`(tweetyenv)/home/you/code/ $ pip install tweetynet`

## Usage
### Training `tweetynet` models to segment and label birdsong
You train `tweetynet` models with the [`vak`](https://github.com/NickleDave/vak) library.
The `vak` library is configured with `config.ini` files, using one of a handful of command-line flags.
As an example, here's how you'd run `vak` from the command line to train a single `config.ini` file:  
`(tweetyenv)$ vak train ./configs/config_bird0.ini`  

For more details, please see the [vak documentation](https://github.com/NickleDave/vak).

### Data and folder structures
To train models, you must supply training data in the form of audio files or 
spectrogram files, and annotations for each spectrogram.

#### Spectrograms and labels
The package can generate spectrograms from `.wav` or `.cbin` audio files.
It can also accept spectrograms in the form of Matlab `.mat` files.

### Important model parameters
* The following parameters must be correctly defined in the configuration `.ini` [file](doc/README_config.md).
  * `n_syllables` - Must be the correct number of labels, including a label for silent periods between syllables.
  * `time_bins` - The number of time bins in each window from the spectrogram shown to the network.
  During training, the network sees batches of windows grabbed randomly from the data whose width are equal to `time_bins`.
  Intuitively, the bigger the time steps, the more temporal context the network has, but the longer it will take 
  to train. In practice, for Bengalese finch song, we achieve good accuracy with 88 time bins, and with canary song, 
  we achieve good accuracy with ~250 time bins. 
   
* The following parameters can be changed if needed:
  * `num_epochs` - Number of times the network should see all the training data.
  * `batch_size` - The number of snippets in each training batch (currently 11)
  * `learning_rate` - The training step rate coefficient (currently 0.001)
Other parameters that specify the network itself can be changed in the code but require knowledge of tensorflow.

## Preparing training files

It is possible to train on any manually annotated data but there are some useful guidelines:
* __Use as many examples as possible__ - The results will just be better. Specifically, this code will not label correctly syllables it did not encounter while training and will most probably generalize to the nearest sample or ignore the syllable.
* __Use noise examples__ - This will make the code very good in ignoring noise.
* __Examples of syllables on noise are important__ - It is a good practice to start with clean recordings. The code will not perform miracles and is most likely to fail if the audio is too corrupt or masked by noise. Still, training with examples of syllables on the background of cage noises will be beneficial.

### Results of running the code

__It is recommended to apply post processing when extracting the actual syllable tag and onset and offset timesfrom the estimates.__

## Predicting new labels

You can predict new labels by adding a [PREDICT] section to the `config.ini` file, and 
then running the command-line interface with the `predict` command, like so:  
`(tweetyenv)$ vak predict ./configs/config_bird0.ini`
An example of what a `config.ini` file with a [PREDICT] section is 
in the doc folder [here](./doc/template_predict.ini).

For users with some scripting / Tensorflow experience, you can
reload a saved model using a checkpoint file saved by the
Tensorflow checkpoint saver. Here's an example of how to do this, taken 
from the `vak.train_utils.learn_curve` function:
```Python
meta_file = glob(os.path.join(training_records_dir, 'checkpoint*meta*'))[0]
data_file = glob(os.path.join(training_records_dir, 'checkpoint*data*'))[0]

model = TweetyNet(n_syllables=n_syllables,
                  input_vec_size=input_vec_size,
                  batch_size=batch_size)

with tf.Session(graph=model.graph) as sess:
    model.restore(sess=sess,
                  meta_file=meta_file,
                  data_file=data_file)
```

## Model architecture
The architecture of this deep neural network is based on these papers:
* S. Böck and M. Schedl, "Polyphonic piano note transcription with recurrent neural networks," 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Kyoto, 2012, pp. 121-124.
doi: 10.1109/ICASSP.2012.6287832 (http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6287832&isnumber=6287775)
* Parascandolo, Huttunen, and Virtanen, “Recurrent Neural Networks for Polyphonic Sound Event Detection in Real Life Recordings.” (https://arxiv.org/abs/1604.00861)

The deep net. structure, used in this code, contains 3 elements:
* 2 convolutional and max pooling layers - A convolutional layer convolves the spectrogram with a set of tunable features and the max pooling is used to limit the number of parameters. These layers allow extracting local spectral and temporal features of syllables and noise.
* A long-short-term-memory recurrent layer (LSTM) - This layer allows the model to incorporate the temporal dependencies in the signal, such as canary trills and the duration of various syllables. The code contains an option to adding more LSTM layers but, since it isn't needed, those are not used.
* A projection layer - For each time bin, this layer projects the previous layer's output on the set of possible syllables. 
