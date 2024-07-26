#!/bin/bash

source .venv/bin/activate

dilation_values=(4 8 16)

for dilation in "${dilation_values[@]}"
do
    echo "Running train.py with dilation=${dilation}"
    python phasefinder/train.py --dilation ${dilation} --model_root "dilationtest_${dilation}"
done

deactivate
