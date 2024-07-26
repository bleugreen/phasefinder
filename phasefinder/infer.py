
import torch
from time import time
from postproc.hmm import hmm_beat_estimation
from postproc.cleaner import clean_beats
from audio.log_filter import create_log_filter, apply_log_filter
import librosa
from model.model import PhasefinderModel
from nnAudio.features import STFT
import torchaudio
import numpy as np
import soundfile as sf
from deeprhythm import DeepRhythmPredictor


N_FFT = 2048
HOP = 512
SAMPLE_RATE = 22050

def make_model(device='cuda'):
    phase_model = PhasefinderModel().to(device)
    phase_model.load_state_dict(torch.load('models/kl-9/kl-9-d8-c36_epoch_16_loss_47153_f_0.854.pt', map_location=torch.device(device)))
    phase_model.eval()
    
    bpm_model = DeepRhythmPredictor('deeprhythm-0.7.pth', device=device)

    filter_matrix = create_log_filter(1025, 81 , device=device)
    stft = STFT(
        n_fft=N_FFT,
        hop_length=HOP,
        sr = SAMPLE_RATE,
        output_format='Magnitude'
        )
    stft = stft.to(device)
       
    return phase_model, bpm_model, stft, filter_matrix

def predict_beats(audio_path, model=None, device='cuda'):
    if model is None:
        phase_model, bpm_model, stft, filter_matrix = make_model()
    else:
        phase_model, bpm_model, stft, filter_matrix = model

    bpm, confidence = bpm_model.predict(audio_path, include_confidence=True)
    
    audio, _ = librosa.load(audio_path, sr=22050)
    audio_tens = torch.tensor(audio).unsqueeze(0).unsqueeze(1).to(device)
    
    stft_batch = torchaudio.functional.amplitude_to_DB(torch.abs(stft(audio_tens)), multiplier=10., amin=0.00001, db_multiplier=1)
    
    song_spec = apply_log_filter(stft_batch, filter_matrix)
    song_spec = (song_spec - song_spec.min()) / (song_spec.max() - song_spec.min())
    
    phase_preds = phase_model(song_spec)
    
    frame_rate = 22050. / 512
    res = hmm_beat_estimation(phase_preds[0, :, :].squeeze(0).to(device), bpm, confidence, frame_rate, device=device)
    bt = torch.tensor(res)
    
    pred_beat_label_onset = torch.abs(bt[1:] - bt[:-1])
    beat_frames = torch.tensor([i for i, x in enumerate(pred_beat_label_onset) if x > 300])
    
    pred_beat_times = beat_frames * 512 / 22050
    cleaned_times = clean_beats(pred_beat_times.numpy())
    
    return cleaned_times, bpm


if __name__ == '__main__':
    model = make_model('cuda')
    audio_path = 'test_songs/deimos.m4a'
    
    start = time()
    beats, bpm = predict_beats(audio_path, model, device='cuda')
    print(bpm)
    print(f'Time: {time()-start:.3f}')
    bigaudio, _ = librosa.load(audio_path, sr=44100)
    click_track = librosa.clicks(times=beats, sr=44100, length=len(bigaudio))
    audio_with_clicks = np.array([click_track, bigaudio])

    # Ensure the audio is in the correct format for saving (transpose if necessary)
    audio_with_clicks = np.vstack([click_track, bigaudio]).T

    # Save the audio file
    sf.write('output_with_clicks.wav', audio_with_clicks, 44100)
    
