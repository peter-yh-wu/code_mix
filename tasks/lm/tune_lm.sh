#!/usr/bin/env bash

## declare an array variable
declare -a momentums=("0.7" "0.5" "0.3")
declare -a learning_rates=("0.001" "0.005" "0.01")
data_path="../SEAME/data/interview/transcript_clean/phaseI"
dp="0.5"
clip="0.25"

## now loop through the above array
for lr in "${learning_rates[@]}"
do
    for mm in "${momentums[@]}"
    do
        config_name="lr_${lr//\./$'v'}_mm_${mm//\./$'v'}_dp_${dp//\./$'v'}_clip_${clip//\./$'v'}"
        echo ""
        echo ""
        echo "============================ ${config_name} ============================"
        echo ""
        echo ""
        mkdir -p "log/${config_name}"
        mkdir -p "models/${config_name}"
        python3 train.py --epoch=15 --data=$data_path --models_dir=models/${config_name} --log_dir=log/${config_name} --lr=${lr} --mm=${mm} --dp=${dp} --clip=${clip}
    done
done