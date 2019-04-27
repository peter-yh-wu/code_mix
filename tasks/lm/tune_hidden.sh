#!/usr/bin/env bash

declare -a hidden_dims=("64" "128" "256" "512")

## declare an array variable
num_epochs=15
data_path="../SEAME/data/interview/transcript_clean/phaseI"
dp="0.5"
clip="0.25"
optim='adam'
gpu_id=1

## now loop through the above array
for hidden in "${hidden_dims[@]}"
do
    config_name="hd_${hidden}_dp_${dp//\./$'v'}_clip_${clip//\./$'v'}"
    echo ""
    echo ""
    echo "============================ ${config_name} ============================"
    echo ""
    echo ""
    mkdir -p "log/${config_name}"
    mkdir -p "models/${config_name}"
    python3 train.py --epoch=${num_epochs} \
                    --data=${data_path} \
                    --models_dir=models/${config_name} \
                    --log_dir=log/${config_name} \
                    --hidden=${hidden} \
                    --dp=${dp} \
                    --clip=${clip} \
                    --optim=${optim} \
                    --gpu_id=${gpu_id}
done