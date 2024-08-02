import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BeatDataset
from postproc.hmm import hmm_beat_estimation
from postproc.cleaner import clean_beats
import librosa
import os
import mir_eval
import numpy as np


def test_model_f_measure(model, data_path, device='cuda'):
    dataset = BeatDataset(data_path, 'test', mode='beat', items=['stft', 'time', 'bpm'], device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    cmlt_scores = []
    amlt_scores = []
    f_measures = []

    with torch.no_grad():
        for stft, beat_times, bpm in tqdm(dataloader):
            stft, beat_times, bpm = stft.to(device), beat_times[0].to(device), bpm.to(device)
            phase_preds = model(stft)
            
            frame_rate = 22050. / 512
            res = hmm_beat_estimation(phase_preds[0][0].to(device), bpm.item(), bpm_confidence=0.9, distance_threshold_factor=0.2, frame_rate=frame_rate, device=device)
            bt = torch.tensor(res)
            pred_beat_label_onset = torch.abs(bt[1:] - bt[:-1])
            beat_frames = np.array([i for i, x in enumerate(pred_beat_label_onset) if x > 300])
            
            pred_beat_times = beat_frames * 512 / 22050
            cleaned_times = clean_beats(pred_beat_times)
            
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

def test_librosa_f_measure(data_path):
    dataset = BeatDataset(data_path, 'test', mode='beat', items=['filepath', 'time', 'bpm'], device='cpu')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    f_measures = []
    cmlt_scores = []
    amlt_scores = []
    with torch.no_grad():
        for filepath, beat_times, bpm  in tqdm(dataloader):
            assert os.path.exists(filepath[0])
            
            audio, _ = librosa.load(filepath[0], sr=22050)
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=22050, bpm=bpm.item())
            np_actual = beat_times.cpu().numpy()[0]
            
            # Convert beat frames to timestamps
            pred_beat_times = librosa.frames_to_time(beat_frames, sr=22050)

            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(np_actual, pred_beat_times)
            f_corr = mir_eval.beat.f_measure(np_actual, pred_beat_times)
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


def test_postprocessing_f_measure(data_path, device='cuda'):
    dataset = BeatDataset(data_path, 'test', mode='beat', items=['stft', 'phase', 'time', 'bpm'], device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    f_measures = []
    cmlt_scores = []
    amlt_scores = []

    with torch.no_grad():
        for stft, phase, beat_times, bpm in tqdm(dataloader):
            stft, phase, beat_times, bpm = stft.to(device), phase.to(device), beat_times[0].to(device), bpm.to(device)
            
            frame_rate = 22050. / 512
            frame_rate = 22050. / 512
            res = hmm_beat_estimation(phase[0][0].to(device), bpm.item(), bpm_confidence=0.9, frame_rate=frame_rate, device=device)
            bt = torch.tensor(res)
            pred_beat_label_onset = torch.abs(bt[1:] - bt[:-1])
            beat_frames = np.array([i for i, x in enumerate(pred_beat_label_onset) if x > 300])
            
            pred_beat_times = beat_frames * 512 / 22050
            cleaned_times = clean_beats(pred_beat_times)
            
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