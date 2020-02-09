# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
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
