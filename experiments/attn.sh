#!/bin/bash

source .venv/bin/activate

use_attention_values=(True False)

for use_attention in "${use_attention_values[@]}"
do
    echo "Running train.py with  use_attention=${use_attention}"
    python phasefinder/train.py --use_attention ${use_attention} --model_root "attn_${use_attention}" --phase_width 7
done

deactivate
