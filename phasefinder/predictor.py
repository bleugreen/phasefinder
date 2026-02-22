from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from deeprhythm import DeepRhythmPredictor
from nnAudio.features import STFT

from phasefinder.audio.log_filter import apply_log_filter, create_log_filter
from phasefinder.constants import (
    BEAT_ONSET_THRESHOLD,
    CLICK_SAMPLE_RATE,
    FRAME_RATE,
    HOP,
    N_FFT,
    SAMPLE_RATE,
)
from phasefinder.model import PhasefinderModelAttn, PhasefinderModelNoattn
from phasefinder.postproc.cleaner import clean_beats
from phasefinder.postproc.hmm import hmm_beat_estimation
from phasefinder.utils import get_device, get_weights


class Phasefinder:
    def __init__(
        self,
        modelname: str = "phasefinder-0.1.pt",
        device: Optional[str] = None,
        quiet: bool = False,
        attention: bool = False,
    ) -> None:
        self.attention = attention
        self.quiet = quiet
        self.model_path = get_weights(modelname, quiet=quiet)
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self.load_model()

    def load_model(self) -> None:
        if self.attention:
            self.model = PhasefinderModelAttn()
        else:
            self.model = PhasefinderModelNoattn()

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device(self.device), weights_only=True),
            strict=False,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.bpm_model = DeepRhythmPredictor("deeprhythm-0.7.pth", device=self.device, quiet=self.quiet)

        fft_bins = int((N_FFT / 2) + 1)
        self.filter_matrix = create_log_filter(fft_bins, 81, device=self.device)
        self.stft = STFT(n_fft=N_FFT, hop_length=HOP, sr=SAMPLE_RATE, output_format="Magnitude")
        self.stft = self.stft.to(self.device)

    def predict(
        self,
        audio_path: Union[str, Path],
        include_bpm: bool = False,
        clean: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        bpm, confidence = self.bpm_model.predict(audio_path, include_confidence=True)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_tens = torch.tensor(audio).unsqueeze(0).unsqueeze(1).to(self.device)

        stft_batch = torchaudio.functional.amplitude_to_DB(
            torch.abs(self.stft(audio_tens)), multiplier=10.0, amin=0.00001, db_multiplier=1
        )

        song_spec = apply_log_filter(stft_batch, self.filter_matrix)
        song_spec = (song_spec - song_spec.min()) / (song_spec.max() - song_spec.min())

        phase_preds = self.model(song_spec)

        res = hmm_beat_estimation(
            phase_preds[0, :, :].squeeze(0).to(self.device),
            bpm,
            bpm_confidence=confidence,
            frame_rate=FRAME_RATE,
            device=self.device,
        )
        bt = torch.tensor(res)

        pred_beat_label_onset = torch.abs(bt[1:] - bt[:-1])
        beat_frames = torch.tensor([i for i, x in enumerate(pred_beat_label_onset) if x > BEAT_ONSET_THRESHOLD])

        pred_beat_times = (beat_frames * HOP / SAMPLE_RATE).numpy()
        if clean:
            pred_beat_times = clean_beats(pred_beat_times)

        if include_bpm:
            return pred_beat_times, bpm
        else:
            return pred_beat_times

    def make_click_track(
        self,
        audio_path: Union[str, Path],
        output_path: str = "output.wav",
        beats: Optional[np.ndarray] = None,
        clean: bool = True,
    ) -> None:
        if beats is None:
            beats = self.predict(audio_path, clean=clean)
        audio, _ = librosa.load(audio_path, sr=CLICK_SAMPLE_RATE)
        click_track = librosa.clicks(times=beats, sr=CLICK_SAMPLE_RATE, length=len(audio))

        audio_with_clicks = np.vstack([click_track, audio]).T

        sf.write(output_path, audio_with_clicks, CLICK_SAMPLE_RATE)
