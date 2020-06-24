# reproduces analysis of models trained on BFSongRepository

# need to make directories for outputs in results
for bird_ID in bl26lb16 gr41rd51 gy6or6 or60yw70
do
  mkdir results/BFSongRepository/${bird_ID}/eval
done

for eval_config in src/configs/BFSongRepository/**/**/*eval*toml
do
  vak prep $eval_config
done

python src/scripts/BFSongRepository/eval_without_and_with_output_transforms.py
