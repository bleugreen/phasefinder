from typing import Union

import numpy as np
import torch

from phasefinder.constants import BEAT_ONSET_THRESHOLD, FRAME_RATE, HOP, SAMPLE_RATE
from phasefinder.postproc.cleaner import clean_beats
from phasefinder.postproc.hmm import hmm_beat_estimation


def extract_beat_times(
    phase_tensor: torch.Tensor,
    bpm: float,
    bpm_confidence: float = 0.9,
    distance_threshold_factor: float = 0.2,
    clean: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Run HMM beat estimation, onset detection, and optional cleaning."""
    res = hmm_beat_estimation(
        phase_tensor,
        bpm,
        bpm_confidence=bpm_confidence,
        distance_threshold_factor=distance_threshold_factor,
        frame_rate=FRAME_RATE,
        device=device,
    )
    bt = torch.tensor(res)
    onset = torch.abs(bt[1:] - bt[:-1])
    beat_frames = np.array([i for i, x in enumerate(onset) if x > BEAT_ONSET_THRESHOLD])
    pred_beat_times = beat_frames * HOP / SAMPLE_RATE
    return clean_beats(pred_beat_times) if clean else pred_beat_times
