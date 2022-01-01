# code to reproduce results in eLife article

This code is provided to reproduce results in the article in eLife
(currently in revision).

## installation

Experiments were run on Ubuntu (>=16.04.07, <22.10), 
using `conda` to isolate the environment.  
We provide environment files to help with reproducibility.  
To work with the code, please first clone this repository 
with `git` and then use those files with `conda` 
to re-create the environment, like so:  

```console
$ git clone https://github.com/yardencsGitHub/tweetynet.git
$ cd tweetynet/article
$ conda create --name article-env --file spec-file.txt
$ conda activate article-env
```

The following commands were used to create a cleaned-up version 
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
