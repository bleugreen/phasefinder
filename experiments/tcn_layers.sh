#!/bin/bash

source .venv/bin/activate

num_layers_values=(8 16 24)

for num_layers in "${num_layers_values[@]}"
do
    echo "Running train.py with num_layers=${num_layers}"
    python phasefinder/train.py --num_tcn_layers ${num_layers} --model_root "tcnlayertest_${num_layers}"
done

deactivate
