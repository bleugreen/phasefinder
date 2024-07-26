#!/bin/bash

source .venv/bin/activate

num_channels_values=(16 24 72)

for num_channels in "${num_channels_values[@]}"
do
    echo "Running train.py with num_channels=${num_channels}"
    python phasefinder/train.py --num_channels ${num_channels} --model_root "chtest_c${num_channels}"
done

deactivate
