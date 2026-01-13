import itertools
import numpy as np
import torch
from model.model_noattn import PhasefinderModelNoattn
from dataset import BeatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from postproc.hmm import hmm_beat_estimation
from postproc.cleaner import clean_beats
import mir_eval
import argparse
import csv
from datetime import datetime
import os
import random

def main(modelname, bpm_confidence=1.0, distance_threshold_factor=0.1, clean_beats_threshold=0.001,
         overlap_threshold=0.3, early_beat_threshold=0.7, late_beat_threshold=1.3, missed_beat_threshold=1.7, mode_threshold=0.001, nudge_amount=0.5):
    datapath = '../stft_db_b_phase_cleaned.h5'

    beat_model = PhasefinderModelNoattn().cuda()
    beat_model.load_state_dict(torch.load(modelname, map_location=torch.device('cuda'), weights_only=True), strict=False)
    beat_model.eval()

    dataset = BeatDataset(datapath, 'test', mode='beat', items=['stft', 'time', 'bpm'], device='cuda')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    f_measures = []
    cmlt_scores = []
    amlt_scores = []

    with torch.no_grad():
        for i, (stft, beat_times, bpm) in enumerate(tqdm(dataloader)):
            stft, beat_times, bpm = stft.to('cuda'), beat_times[0].to('cuda'), bpm.to('cuda')
            phase_preds = beat_model(stft)
            
            frame_rate = 22050. / 512
            res = hmm_beat_estimation(phase_preds[0][0].to('cuda'), bpm.item(), frame_rate, 
                                      bpm_confidence=bpm_confidence, 
                                      distance_threshold_factor=distance_threshold_factor, 
                                      device='cuda')
            bt = torch.tensor(res)
            pred_beat_label_onset = torch.abs(bt[1:] - bt[:-1])
            beat_frames = np.array([i for i, x in enumerate(pred_beat_label_onset) if x > 300])
            
            pred_beat_times = beat_frames / frame_rate
            cleaned_times = clean_beats(pred_beat_times, 
                                        clean_beats_threshold=clean_beats_threshold,
                                        overlap_threshold=overlap_threshold,
                                        early_threshold=early_beat_threshold,
                                        late_threshold=late_beat_threshold,
                                        missed_threshold=missed_beat_threshold,
                                        mode_threshold=mode_threshold,
                                        nudge_amount=nudge_amount)
            
            np_actual = beat_times.cpu().numpy()
            np_pred = cleaned_times
            
            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(np_actual, np_pred)
            f_corr = mir_eval.beat.f_measure(np_actual, np_pred)

            f_measures.append(f_corr)
            cmlt_scores.append(cmlt)
            amlt_scores.append(amlt)

    overall_f_measure = sum(f_measures) / len(f_measures)    
    print(f"Overall F-measure: {overall_f_measure:.3f}")
    overall_cmlt = sum(cmlt_scores) / len(cmlt_scores)
    overall_amlt = sum(amlt_scores) / len(amlt_scores)
    print(f"Overall CMLt: {overall_cmlt:.3f}")
    print(f"Overall AMLt: {overall_amlt:.3f}")

    return overall_f_measure, overall_cmlt, overall_amlt

def read_existing_results(filename):
    tested_combinations = set()
    best_f_measure = 0
    best_params = None
    
    if os.path.exists(filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                params = tuple(map(float, row[:7]))  # Convert first 7 elements to float
                tested_combinations.add(params)
                f_measure = float(row[7])
                if f_measure > best_f_measure:
                    best_f_measure = f_measure
                    best_params = params
    
    return tested_combinations, best_f_measure, best_params

def grid_search(input_csv=None):
    # Define parameter ranges
    bpm_confidences = [0.9]  # [0.1, 0.5, 0.7, 0.9]
    distance_threshold_factors = [0.15, 0.2, 0.25]
    clean_beats_thresholds = [0.03, 0.05, 0.1]
    overlap_thresholds = [ 0.15, 0.2,0.25, 0.35, 0.4]
    early_beat_thresholds = [0.55, 0.6, 0.65]
    late_beat_thresholds = [1.3,1.35, 1.4]
    missed_beat_thresholds = [1.6,1.65,1.8]
    mode_thresholds = [0.01,0.04,0.08, 0.1]
    nudge_amounts = [0.45, 0.5, 0.55]


    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        bpm_confidences, distance_threshold_factors, clean_beats_thresholds,
        overlap_thresholds, early_beat_thresholds, late_beat_thresholds, missed_beat_thresholds, mode_thresholds, nudge_amounts
    ))

    

    # If input CSV is provided, read existing results
    if input_csv:
        tested_combinations, best_f_measure, best_params = read_existing_results(input_csv)
        param_combinations = [combo for combo in param_combinations if combo not in tested_combinations]
        results_filename = input_csv
    else:
        best_f_measure = 0
        best_params = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"grid_search_results_{timestamp}.csv"

        # Write header to the CSV file if it's a new file
        with open(results_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['bpm_confidence', 'distance_threshold_factor', 'clean_beats_threshold',
                             'overlap_threshold', 'early_beat_threshold', 'late_beat_threshold',
                             'missed_beat_threshold','mode_threshold', 'nudge_amount', 'f_measure', 'cmlt', 'amlt'])
    random.shuffle(param_combinations)
    print(f'num combos = {len(param_combinations)}')
    for params in param_combinations:
        print(f"Parameters: {params}")
        f_measure, cmlt, amlt = main('../phasefinder-0.1-noattn.pt', *params)

        # Append results to the CSV file
        with open(results_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list(params) + [f_measure, cmlt, amlt])

        if f_measure > best_f_measure:
            best_f_measure = f_measure
            best_params = params
        print("--------------------")

    print(f"Best parameters: {best_params}")
    print(f"Best F-measure: {best_f_measure}")
    print(f"Results saved to: {results_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run grid search for beat estimation parameters')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file with existing results', default=None)
    args = parser.parse_args()
    
    grid_search(args.input_csv)