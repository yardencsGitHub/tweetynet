#!/bin/bash

# must run this from ./tweetynet/article
# after following set-up directions in ./tweetynet/article/README.md
ARTICLE_ROOT=$PWD
RESULTS_DIR=${ARTICLE_ROOT}/RESULTS
CONFIGS=${ARTICLE_ROOT}/data/configs/ReplicateKeyResult/*toml

for config in ${CONFIGS}:
  vak prep ${config}
  vak eval ${config}
