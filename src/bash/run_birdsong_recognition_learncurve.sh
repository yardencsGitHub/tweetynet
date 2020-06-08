# generate learning curve for all birds in BirdsongRecognition dataset
for config in src/configs/BirdsongRecognition/*.toml;
do
  vak prep $config
  vak learncurve $config
done
