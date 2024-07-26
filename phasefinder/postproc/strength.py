import numpy as np
from scipy.signal import spectrogram
import librosa 

def beat_estimation_strength(audio, sample_rate, beat_times):
    def safe_normalize(x):
        if np.all(x == x[0]):
            return np.ones_like(x)  
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    # Input validation
    if len(audio) == 0 or len(beat_times) == 0:
        print("Error: Empty audio or beat_times")
        return None

    # 1. Consistency of inter-beat intervals
    intervals = np.diff(beat_times)
    if len(intervals) == 0:
        print("Error: Not enough beat times to calculate intervals")
        return None
    interval_consistency = 1 / (np.std(intervals) + 1e-6)

    # 2. Alignment with audio energy peaks
    window_size = int(0.05 * sample_rate)
    energy_at_beats = np.mean([np.sum(audio[max(0, int(b*sample_rate)):min(len(audio), int(b*sample_rate)+window_size)]**2) 
                               for b in beat_times])

    # 3. Spectral flux at beat times
    _, _, Sxx = spectrogram(audio, sample_rate)
    if Sxx.size == 0:
        print("Error: Empty spectrogram")
        return None
    spectral_flux = np.sum(np.diff(Sxx, axis=1), axis=0)
    flux_at_beats = np.mean([spectral_flux[min(int(b*sample_rate/len(spectral_flux)), len(spectral_flux)-1)] for b in beat_times])

    # 4. Tempo stability
    tempos = 60 / intervals
    tempo_stability = 1 / (np.std(tempos) + 1e-6)

    # 5. Rhythmic pattern repetition
    pattern_length = 4 
    patterns = [intervals[i:i+pattern_length] for i in range(0, len(intervals)-pattern_length, pattern_length)]
    if len(patterns) > 1:
        pattern_similarities = [np.corrcoef(patterns[i], patterns[i+1])[0,1] for i in range(len(patterns)-1)]
        pattern_similarity = np.mean([ps for ps in pattern_similarities if not np.isnan(ps)])
    else:
        pattern_similarity = 0

    # 6. Low-frequency content alignment
    low_freq_energy = np.sum(Sxx[:max(1, int(len(Sxx)/4))], axis=0)  # Use lower quarter of spectrum
    low_freq_at_beats = np.mean([low_freq_energy[min(int(b*sample_rate/len(low_freq_energy)), len(low_freq_energy)-1)] for b in beat_times])

    # 7. Onset detection correlation
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    onset_at_beats = np.mean([onset_env[min(int(b*len(onset_env)/len(audio)), len(onset_env)-1)] for b in beat_times])

    # Normalize and combine all factors
    factors = [
        interval_consistency,
        energy_at_beats,
        flux_at_beats,
        tempo_stability,
        pattern_similarity,
        low_freq_at_beats,
        onset_at_beats
    ]
    
    normalized_factors = safe_normalize(np.array(factors))
    
    # You can adjust these weights based on their importance
    weights = [2, 0.5, 1, 1, 0.5, 2, 0.5]
    
    result = np.average(normalized_factors, weights=weights)
    
    return result