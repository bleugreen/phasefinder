#!/bin/bash

source .venv/bin/activate
shared_flags="--phase_width 7 --max_epochs 40 --lr 2e-4 --data_path stft_db_b_phase_cleaned.h5 --start_epoch 19"
echo "Running train.py with use_attention=True"
python phasefinder/train.py --use_attention True --model_root "attn_True_ext" --load_weights attn_True_epoch_18_loss_46979.8633_f_0.816_cmlt_0.785_amlt_0.875.pt $shared_flags
echo "Running train.py with use_attention=False"
python phasefinder/train.py --use_attention False --model_root "attn_False_ext" --load_weights attn_False2_epoch_18_loss_47182.0596_f_0.834_cmlt_0.804_amlt_0.873.pt $shared_flags
deactivate
