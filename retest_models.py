import os
import re
import torch
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import sys
import librosa
sys.path.append('/home/bleu/ai/phasefinder/src')
from phasefinder.model.model_attn import PhasefinderModelAttn
from phasefinder.val import test_model_f_measure
import argparse

def recalculate_and_move_checkpoints(source_dir, dest_dir, data_path='stft_db_b_phase_cleaned.h5'):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Regular expression to match the filename format
    pattern = r'kl9-pw(\d+)_epoch_(\d+)_loss_([\d.]+)_f_([\d.]+)_cmlt_([\d.]+)_amlt_([\d.]+)\.pt'

    for filename in os.listdir(source_dir):
        match = re.match(pattern, filename)
        if match:
            pw, epoch, loss, _, _, _ = match.groups()
            pw = int(pw)
            epoch = int(epoch)
            loss = int(float(loss))

            # Load the model
            model_path = os.path.join(source_dir, filename)
            print(filename, pw, epoch, loss)
            beat_model = PhasefinderModelAttn().cuda()
            beat_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
            beat_model.eval()

            # Recalculate F-measure, CMLT, and AMLT
            new_f, new_cmlt, new_amlt = test_model_f_measure(beat_model, data_path)

            # Create new filename
            new_filename = f'kl9-pw{pw}_epoch_{epoch}_loss_{loss}_f_{new_f:.3f}_cmlt_{new_cmlt:.3f}_amlt_{new_amlt:.3f}.pt'

            # Move and rename the file
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, new_filename)
            shutil.move(src_path, dest_path)

            print(f"Processed: {filename} -> {new_filename}")

recalculate_and_move_checkpoints('moved_models/', 'moved2/')