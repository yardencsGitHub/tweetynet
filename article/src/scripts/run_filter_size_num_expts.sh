#!/bin/bash

ANIMAL_IDS=bl26lb16 gr41rd51 gy6or6 or60yw70

for filter_size in 3 7; do
  for animal_id in $ANIMAL_IDS; do
    vak learncurve data/configs/Bengalese_Finches/filter_size/filter_size_${filter_size}/config_BFSongRepository_${animal_id}_learncurve.toml
  done
done

for filter_num in 16 64; do
  for animal_id in $ANIMAL_IDS; do
    vak learncurve data/configs/Bengalese_Finches/filter_num/filter_num_${filter_num}/config_BFSongRepository_${animal_id}_learncurve.toml
  done
done
