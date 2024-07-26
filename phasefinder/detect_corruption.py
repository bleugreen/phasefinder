import h5py
import numpy as np
import os
from tqdm import tqdm

def compute_noise_features(stft):
    avg_spectrum = np.mean(np.abs(stft), axis=1)
    
    # Spectral Flatness
    geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
    arithmetic_mean = np.mean(avg_spectrum)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    # High Frequency Ratio
    high_freq_start = int(0.7 * len(avg_spectrum))
    high_freq_ratio = np.sum(avg_spectrum[high_freq_start:]) / np.sum(avg_spectrum)
    
    return spectral_flatness, high_freq_ratio

def detect_noisy_tracks(file_path, group, flatness_threshold=0.9985, high_freq_threshold=0.275):
    with h5py.File(file_path, 'r') as file:
        noisy_tracks = []
        for track in tqdm(file[group].keys(), desc=f"Analyzing {group}"):
            stft = file[group][track]['stft'][:]
            flatness, high_freq_ratio = compute_noise_features(stft)
            if flatness > flatness_threshold and high_freq_ratio > high_freq_threshold:
                noisy_tracks.append(track)
    return noisy_tracks

def remove_noisy_tracks(file_path, flatness_threshold=0.9985, high_freq_threshold=0.275):
    with h5py.File(file_path, 'a') as h5_file:
        # Identify noisy tracks in all groups
        noisy_tracks = {}
        for group in ['train', 'test', 'val']:
            if group in h5_file:
                noisy_tracks[group] = detect_noisy_tracks(file_path, group, flatness_threshold, high_freq_threshold)
                print(f"Detected {len(noisy_tracks[group])} noisy tracks in {group} group")

        # Delete noisy tracks
        for group in ['train', 'test', 'val']:
            if group in h5_file:
                group_noisy = set(noisy_tracks[group])
                group_ref = h5_file[group]
                
                for track in tqdm(group_noisy, desc=f"Removing noisy tracks from {group}"):
                    del group_ref[track]

    print(f"Cleaned dataset saved to {file_path}")

    
# Verify the cleaning process
def count_tracks(file_path):
    with h5py.File(file_path, 'r') as f:
        return {group: len(f[group]) for group in ['train', 'test', 'val'] if group in f}


if __name__ == '__main__':
    # Example usage
    file_path = 'stft_db_b_phase.hdf5'
    print("\Original dataset:")
    print(count_tracks(file_path))

    remove_noisy_tracks(file_path)

    print("\nCleaned dataset:")
    print(count_tracks(file_path))
