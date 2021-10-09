#!/bin/bash
# run this script to generate source data for all error curves,
# for the main results
# and for the results of hyperparameter experiments


python src/scripts/make_error_curve_with_and_without_output_transforms.py \
  --results-root results/Bengalese_Finches \
  --min-segment-dur-ini ./data/configs/min_segment_dur.ini
