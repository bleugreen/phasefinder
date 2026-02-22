from typing import Tuple

import librosa
import mir_eval
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from phasefinder.dataset import BeatDataset
from phasefinder.postproc import extract_beat_times


def _report_scores(
    f_measures: list[float],
    cmlt_scores: list[float],
    amlt_scores: list[float],
) -> Tuple[float, float, float]:
    """Average and print evaluation scores."""
    overall_f_measure = sum(f_measures) / len(f_measures)
    overall_cmlt = sum(cmlt_scores) / len(cmlt_scores)
    overall_amlt = sum(amlt_scores) / len(amlt_scores)
    print(f"Overall F-measure: {overall_f_measure:.3f}")
    print(f"Overall CMLt: {overall_cmlt:.3f}")
    print(f"Overall AMLt: {overall_amlt:.3f}")
    return overall_f_measure, overall_cmlt, overall_amlt


def test_model_f_measure(
    model: torch.nn.Module,
    data_path: str,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    dataset = BeatDataset(data_path, "test", mode="beat", items=["stft", "time", "bpm"], device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    cmlt_scores: list[float] = []
    amlt_scores: list[float] = []
    f_measures: list[float] = []

    with torch.no_grad():
        for stft, beat_times, bpm in tqdm(dataloader):
            stft, beat_times, bpm = stft.to(device), beat_times[0].to(device), bpm.to(device)
            phase_preds = model(stft)

            cleaned_times = extract_beat_times(
                phase_preds[0][0].to(device),
                bpm.item(),
                bpm_confidence=0.9,
                distance_threshold_factor=0.2,
                clean=True,
                device=device,
            )

            np_actual = beat_times.cpu().numpy()

            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(np_actual, cleaned_times)
            f_corr = mir_eval.beat.f_measure(np_actual, cleaned_times)

            f_measures.append(f_corr)
            cmlt_scores.append(cmlt)
            amlt_scores.append(amlt)

    return _report_scores(f_measures, cmlt_scores, amlt_scores)


def test_librosa_f_measure(data_path: str) -> Tuple[float, float, float]:
    dataset = BeatDataset(data_path, "test", mode="beat", items=["filepath", "time", "bpm"], device="cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    f_measures: list[float] = []
    cmlt_scores: list[float] = []
    amlt_scores: list[float] = []
    with torch.no_grad():
        for filepath, beat_times, bpm in tqdm(dataloader):
            assert os.path.exists(filepath[0])

            audio, _ = librosa.load(filepath[0], sr=22050)
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=22050, bpm=bpm.item())
            np_actual = beat_times.cpu().numpy()[0]

            pred_beat_times = librosa.frames_to_time(beat_frames, sr=22050)

            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(np_actual, pred_beat_times)
            f_corr = mir_eval.beat.f_measure(np_actual, pred_beat_times)
            f_measures.append(f_corr)
            cmlt_scores.append(cmlt)
            amlt_scores.append(amlt)

    return _report_scores(f_measures, cmlt_scores, amlt_scores)


def test_postprocessing_f_measure(
    data_path: str,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    dataset = BeatDataset(data_path, "test", mode="beat", items=["stft", "phase", "time", "bpm"], device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    f_measures: list[float] = []
    cmlt_scores: list[float] = []
    amlt_scores: list[float] = []

    with torch.no_grad():
        for stft, phase, beat_times, bpm in tqdm(dataloader):
            stft, phase, beat_times, bpm = stft.to(device), phase.to(device), beat_times[0].to(device), bpm.to(device)

            cleaned_times = extract_beat_times(
                phase[0][0].to(device),
                bpm.item(),
                bpm_confidence=0.9,
                clean=True,
                device=device,
            )

            np_actual = beat_times.cpu().numpy()

            cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(np_actual, cleaned_times)
            f_corr = mir_eval.beat.f_measure(np_actual, cleaned_times)

            f_measures.append(f_corr)
            cmlt_scores.append(cmlt)
            amlt_scores.append(amlt)

    return _report_scores(f_measures, cmlt_scores, amlt_scores)
