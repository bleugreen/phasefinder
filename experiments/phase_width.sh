#!/bin/bash

source .venv/bin/activate

pw_values=(3 5 7 9 11)

for pw in "${pw_values[@]}"
do
    echo "Running train.py with phase_width=${pw}"
    python phasefinder/train.py --phase_width ${pw} --model_root "pwtest_${pw}"
done

deactivate
