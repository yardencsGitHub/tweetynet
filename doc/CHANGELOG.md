# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.7.0 -- 2021-01-05
### Added
- add code related to eLife article.
  See [revisions project](https://github.com/yardencsGitHub/tweetynet/projects/1) for details.

### Changed
- make many code changes related to eLife article.
  See [revisions project](https://github.com/yardencsGitHub/tweetynet/projects/1) for details.
- refactor project so code for article is separate from `tweetynet` code 
  [#188](https://github.com/yardencsGitHub/tweetynet/pull/56)
  Fixes [#187](https://github.com/yardencsGitHub/tweetynet/issues/187).
- raise minimum required versions of `vak` and `pytorch`
  [9708e2b](https://github.com/yardencsGitHub/tweetynet/commit/9708e2bab2b3ddb175c893396a40670a5da82b23).

## [0.6.0](https://github.com/yardencsGitHub/tweetynet/releases/tag/0.6.0) -- 2021-04-04
### Added
- add hyperparameters for RNN component of `TweetyNet` architecture
  to its `__init__` function, including size of hidden state
  [#73](https://github.com/yardencsGitHub/tweetynet/pull/73).
  Fixes [#70](https://github.com/yardencsGitHub/tweetynet/issues/70).

## [0.5.0]
### Changed
- rewrite README.md 
  [#56](https://github.com/yardencsGitHub/tweetynet/pull/56)
- rename `yarden2annot` parameter `annot_file` to `annot_path` 
  to work with `crowsetta` version 3.0.0

### Fixed
- fix script for running `eval` on `BFSongRepo` models
  [#49](https://github.com/yardencsGitHub/tweetynet/pull/49)
- fix bug in BFRepository-results/all-birds.ipynb -- wrong variable name

## [0.4.4]
### Changed
- specify minimum required version of `vak` as 0.3.1 

## [0.4.3]
### Changed
- translate config files in `./src/configs` to `vak 0.3.0` format
- specify minimum required version of `vak` as 0.3.0 

## [0.4.2]
### Added
- add Levenshtein distance and segment error rate to metrics
  [#37](https://github.com/yardencsGitHub/tweetynet/pull/37)

## [0.4.1]
### Fixed
- change optimizer back to Adam, should not have been SGD
  [#36](https://github.com/yardencsGitHub/tweetynet/pull/36)

## [0.4.0]
### Added
- add `logger` parameter to `TweetyNetModel.from_config` class method
  [#34](https://github.com/yardencsGitHub/tweetynet/pull/34)
  + to take advantage of `logger` attribute being added to `vak.Model` -- i.e. be able to log training

## [0.3.1]
### Changed
- make `yarden2annot` return labels as strings [#32](https://github.com/yardencsGitHub/tweetynet/pull/32)
  + to be consistent with what `vak` expects

## [0.3.0]
### Added
- `network.py` module (see "Changed")
- initial `tests/test_tweetynet.py` module, written for `pytest`

### Changed
- convert to `PyTorch` model
  + because Tensorflow 1.0 is deprecated.
  + chose PyTorch over Tensorflow 2.0 for several reasons, see 
    [NickleDave/vak#88](https://github.com/NickleDave/vak/pull/88)
  + network is now implemented as `TweetyNet` class in `network.py`,
    as a subclass of `torch.nn.Module`
- make `vak` a dependency, since we use `vak.Model`
    - rewrite `model.py` with `TweetyNetModel`, a `vak.Model` subclass
      + that specifies optimizer, loss, and metrics for `TweetyNetModel`
- `yarden2seq` re-written to work with `crowsetta 2.0`, renamed `yarden2annot`

## [0.2.0]
### Added
- add `gardner` package with `yarden2seq` module that converts `annotation.mat` files to 
  `crowsetta.Sequence` objects
  + will be installed along with TweetyNet so that `vak` knows how to use `yarden` format

## [0.1.1a4]
### Changed
- change README.md to state that TweetyNet can be run with `vak` + point to the `vak` docs

### Removed
- remove `teweetynet/graphs.py` that was no longer being used
- remove tweetynet-cli script + `console_script` entry point
  + old version of script no longer worked with `vak`
  + tried just using `vak` cli but this can give user confusing error messages where `vak` and
  `tweetynet` are both mentioned; getting that to work cleanly seems like more trouble than its worth

## [0.1.1a3]
### Changed
- Change dependency to `vak` (<https://github.com/NickleDave/vak>) after
renaming `songdeck` to `vak`--shouldn't affect usage of TweetyNet, but need to 
release new version so dependencies work when installing from PyPI

## [0.1.1a2]
### Changed
- Version where network is abstracted away from code for benchmarking;
  put that code in a separate library, `songdeck`

## [0.1.1a1]
- Original version uploaded to PyPI
