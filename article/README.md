# code to reproduce results in eLife article

This code is provided to reproduce results in the article in eLife:
https://elifesciences.org/articles/63853

## installation

Experiments were run on Ubuntu (>=16.04.07, <22.10), 
using `conda` to isolate the environment.  
We provide environment files to help with reproducibility.  
To work with the code, please first clone this repository 
with `git` and then use those files with `conda` 
to re-create the environment, like so:  

### 1. clone the repository

```console
$ git clone https://github.com/yardencsGitHub/tweetynet.git
```

### 2. use the `conda` environment files to recreate the environment

```console
$ cd tweetynet/article
$ conda create --name article-env --file spec-file.txt
$ conda activate article-env
```

### 3. install the code used for the paper into the environment

```
(article-env) $ cd ./tweetynet/article
(article-env) $ pip install .
```

> The following commands were used to create a cleaned-up version 
of the environment, and then generate files from that environment. 
(These commands may not be reproducible after newer versions of 
the dependencies are released.)
```console
$  conda create -n article-env python=3.8
$  conda activate article-env
$  conda install pytorch==1.7.1 torchvision==0.8.2 -c pytorch
$  pip install .
$ conda env export > environment.yml
$ conda list --explicit > spec-file.txt
```

### 4. download datasets

The datasets linked in the manuscript are under review at Dryad.
We provide links to the data in cloud storage as an alternate source. 

#### Annotated canary song dataset

Audio files, annotations, and spectrograms for song from 3 canaries can be downloaded here:  
https://www.dropbox.com/sh/3lmcrokcgsctld3/AACW0gLZpmfNU-ukq68p8CSfa?dl=0

#### Small dataset of saved model checkpoints

These are checkpoints that achieved the lowest error
on the benchmark datasets used,
as reported in the Results section in the manuscript titled 
"TweetyNet annotates with low error rates across individuals and species".

https://drive.google.com/drive/folders/1slQ7RzZn9ZnibvWb9ut37AMgpToMmm8x?usp=sharing

## usage

To replicate the error rates obtained by the checkpoints we provide, 
run the script:

```console
(article-env) $  src/scrips/replicate-key-result/runall.sh
```

(This script is a work-in-progess, will be completed in this issue:
https://github.com/yardencsGitHub/tweetynet/issues/197)
