[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2667812.svg)](https://doi.org/10.5281/zenodo.2667812)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![PyPI version](https://badge.fury.io/py/tweetynet.svg)](https://badge.fury.io/py/tweetynet)

# TweetyNet
<p align="center"><img src="./doc/tweetynet.gif" alt="tweetynet image" width=100></p>

## What is `tweetynet`?
A neural network architecture (shown below) 
that automates annotation of birdsong and other vocalizations by segmenting spectrograms, 
and then labeling those segments.  
<p align="center">
<img src="article/doc/figures/mainfig_tweetynet_architecture_and_basic_operation/mainfig_tweetynet_architecture_operations_and_post_processing.png" alt="neural network architecture" width=600>
</p>

This is an example of the kind of annotations that `tweetynet` learns to predict:  
<p align="center">
<img src="doc/annotation-example.png" width=350>
</p>

## How is it used?
### Installation
#### To train models and use them to predict annotation
To install, run the following command at the command line:  
`pip install tweetynet`
#### To train models and use them to predict annotation
To facilitate training `tweetynet` models and using trained models 
to predict annotation on new datasets, 
we developed the `vak` library, 
that is installed automatically with `tweetynet`.

Please see the `vak` documentation for detailed installation instructions:  
https://vak.readthedocs.io/en/latest/get_started/installation.html  

A link to a tutorial on using `tweetynet` with `vak` is below.

#### To reproduce results from article
In the directory [`./article`](./article) 
we provide code to reproduce the results in the article   
"TweetyNet: A neural network that enables high-throughput, automated annotation of birdsong"  
https://www.biorxiv.org/content/10.1101/2020.08.28.272088v2
(in revision at eLife).

Please see the [README](./article/README.md) in that directory 
for instructions on how to install and work with that code.

#### General use

### Usage
For a tutorial on using `tweetynet` with `vak`, please see the `vak` documentation:  
https://vak.readthedocs.io/en/latest/tutorial/autoannotate.html

### FAQs
#### Training data
To train models, you must supply training data in the form of audio files or 
spectrogram files, and annotations.
The package can generate spectrograms from `.wav` or `.cbin` audio files.
It can also accept spectrograms in the form of Matlab `.mat` files or `.npz` files created by `numpy`.
`vak` uses a separate library to parse annotations, `crowsetta`, 
which handles some common formats and can also be used to write custom parsers for other formats.
Please see the `crowsetta` documentation for more detail:  
https://crowsetta.readthedocs.io/en/latest/#

#### Preparing training files
It is possible to train on any manually annotated data but there are some useful guidelines:
* __Use as many examples as possible__ - The results will just be better. Specifically, this code will not label correctly syllables it did not encounter while training and will most probably generalize to the nearest sample or ignore the syllable.
* __Use noise examples__ - This will make the code very good in ignoring noise.
* __Examples of syllables on noise are important__ - It is a good practice to start with clean recordings. The code will not perform miracles and is most likely to fail if the audio is too corrupt or masked by noise. Still, training with examples of syllables on the background of cage noises will be beneficial.

For more details, please see the [vak documentation](https://github.com/NickleDave/vak).

## Issues
If you run into problems, please use the [issue tracker](https://github.com/yardencsGitHub/tweetynet/issues) 
or contact the authors via email in the paper above.

## Citation
If you use or adapt this code, please cite its DOI:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2667812.svg)](https://doi.org/10.5281/zenodo.2667812)

## License
Released under [BSD license](./LICENSE).


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://yardencsgithub.github.io/"><img src="https://avatars.githubusercontent.com/u/17324841?v=4?s=100" width="100px;" alt=""/><br /><sub><b>yardencsGitHub</b></sub></a><br /><a href="https://github.com/yardencsGitHub/tweetynet/commits?author=yardencsGitHub" title="Code">💻</a> <a href="https://github.com/yardencsGitHub/tweetynet/issues?q=author%3AyardencsGitHub" title="Bug reports">🐛</a> <a href="#data-yardencsGitHub" title="Data">🔣</a> <a href="https://github.com/yardencsGitHub/tweetynet/commits?author=yardencsGitHub" title="Documentation">📖</a> <a href="#ideas-yardencsGitHub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-yardencsGitHub" title="Answering Questions">💬</a> <a href="#tool-yardencsGitHub" title="Tools">🔧</a> <a href="https://github.com/yardencsGitHub/tweetynet/commits?author=yardencsGitHub" title="Tests">⚠️</a> <a href="#tutorial-yardencsGitHub" title="Tutorials">✅</a> <a href="#talk-yardencsGitHub" title="Talks">📢</a></td>
    <td align="center"><a href="https://nicholdav.info/"><img src="https://avatars.githubusercontent.com/u/11934090?v=4?s=100" width="100px;" alt=""/><br /><sub><b>David Nicholson</b></sub></a><br /><a href="https://github.com/yardencsGitHub/tweetynet/commits?author=NickleDave" title="Code">💻</a> <a href="https://github.com/yardencsGitHub/tweetynet/issues?q=author%3ANickleDave" title="Bug reports">🐛</a> <a href="#data-NickleDave" title="Data">🔣</a> <a href="https://github.com/yardencsGitHub/tweetynet/commits?author=NickleDave" title="Documentation">📖</a> <a href="#ideas-NickleDave" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-NickleDave" title="Answering Questions">💬</a> <a href="#tool-NickleDave" title="Tools">🔧</a> <a href="https://github.com/yardencsGitHub/tweetynet/commits?author=NickleDave" title="Tests">⚠️</a> <a href="#tutorial-NickleDave" title="Tutorials">✅</a> <a href="#talk-NickleDave" title="Talks">📢</a></td>
    <td align="center"><a href="https://github.com/zhehao-nkd"><img src="https://avatars.githubusercontent.com/u/45915756?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zhehao Cheng</b></sub></a><br /><a href="https://github.com/yardencsGitHub/tweetynet/issues?q=author%3Azhehao-nkd" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!