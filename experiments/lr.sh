#!/bin/bash

source .venv/bin/activate

lr_values=(2e-4 5e-4 1e-3 2e-3)

for lr in "${lr_values[@]}"
do
    echo "Running train.py with learning rate=${num_channels}"
    python phasefinder/train.py --lr ${lr} --model_root "lrtest_${lr}"
done

deactivate