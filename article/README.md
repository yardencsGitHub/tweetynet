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
$ cd ./tweetynet/article
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
https://github.com/yardencsGitHub/tweetynet/issues/211)

### workflow for generating final results

Here we document how we produced final results in the paper. 
We do this because we cannot easily share all of the results files, 
that taken together are roughly ~0.5 TB. 
For this reason we do not include all files in the "results" dataset  
that accompanies the article. 
Please feel free to contact the authors 
with questions about specific results.
We have compressed most of the results into .tar.gz files 
that we can share upon request, 
using the scripts in `./src/scripts/make-results-datasets`.

#### 1. Run experiments with `vak` and `hvc` libraries
1. Run experiments with `tweetynet` with .toml configuration files for `vak` in `./data/configs`.
The vast majority use the `learncurve` functionality, e.g. for results in figures 3, 4, and 5.

```console
(article-env) $ vak learncurve ./data/configs/Bengalese_Finches/learncurve/
```

2. Run experiments with Support Vector Machine model, by running `src/scripts/tweetynet-v-svm/run_learncurve.py`.
This script expects that the `vak learncurve` experiments have already been run, 
since it uses the exact same splits. The script uses the `article` package and code from the 
`hybrid-vocal-classifier` (`hvc`) library that was installed when creating the environment as above.

For example, to run the experiments for Bengalese finch song: 
```console
(article-env) $ python src/scripts/tweetynet-v-svm/run_learncurve.py \
    --results_root ./results/Bengalese_Finches\learncurve \
    --animal_ids bl26lb16 gr41rd51 gy6or6 or60yw70 \
    --segment_params_ini ./data/configs/segment_params.ini \
    --csv-filename segment_error_across_birds.hvc.csv \
    --results_dst ./results/Bengalese_Finches/hvc
```

#### 2. Generate summary .csv files
1. For each of the experiments, run `./src/scripts/make_error_curve_with_and_without_output_transforms.py`.
This script produces the .csv files named `error_across_birds_with_cleanup.csv` 
that are in the subdirectories of `./results`.

You will need to run the script once for each experiment.
Experiments are divided up by species and by the condition we were testing.
The `--results-root` argument should point to the species name in `./results`,
and the `--experiment-dirs` argument to the script should point to a directory
that contains results from individual animals, like so:
```console
results/Bengalese_Finches  # <-- you want to specify this path as `results-root` argument when calling this script
└── learncurve  # <-- you want to specify these sub-directories as 'experiment-dirs'
    ├── bird1  # <-- animal IDs
    │   └──results_210409_092101  # <-- results directories generated by runs of `vak learncurve`
    ├── bird2
    │   └──results_210409_011292
    ├ ...
```

Note that there are multiple sub-directories for some experiments.

E.g., for the main results in Figure 4 we ran all the config files in `Bengalese_Finches/learncurve`
and `Canaries/learncurve`. To replicate this run:
```console
(article-env) $ python ./src/scripts/make_error_curve_with_and_without_output_transforms.py \
    --results-root .results/Bengalese_Finches \
    --experiment_dirs learncurve \
    --segment_params_ini ./data/configs/segment_params.ini \
    --csv-filename error_across_birds_with_cleanup.csv \
(article-env) $ python ./src/scripts/make_error_curve_with_and_without_output_transforms.py \
    --results-root .results/canaries \
    --experiment_dirs learncurve \
    --segment_params_ini ./data/configs/segment_params.ini \
    --csv-filename error_across_birds_with_cleanup.csv \
```

E.g., for the experiments testing the effect of the hidden size of the recurrent layer of TweetyNet,
you would need to run the script once for `./results/Bengalese_Finches/hidden_size/hidden_size_64`,
and again for `hidden_size/hidden_size_16`.
```console
(article-env) $ python ./src/scripts/make_error_curve_with_and_without_output_transforms.py \
    --results-root .results/Bengalese_Finches \
    --experiment_dirs learncurve hidden_size/hidden_size_16 hidden_size/hidden_size_64 \
    --segment_params_ini ./data/configs/segment_params.ini \
    --csv-filename error_across_birds_with_cleanup.csv \
```

#### 3. Generate source data files for figures
1. Finally run `src/scripts/generate_figure_source_data.py`

This script uses the final results to generate source data .csv files for each figure.
Those .csv files are saved in `./doc/figures` and loaded by the Jupyter notebooks 
in `./src/scripts` that plot the figures.

```console
(article-env) $ python ./src/scripts/generate_figure_source_data.py
```