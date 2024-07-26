import torch
import numpy as np

def create_log_filter(num_bins, num_bands, device='cuda'):
    log_bins = np.logspace(np.log10(1), np.log10(num_bins), num=num_bands+1, base=10) - 1
    log_bins = np.floor(log_bins).astype(int)
    log_bins[-1] = num_bins  
    if len(np.unique(log_bins)) < len(log_bins):
        for i in range(1, len(log_bins)):
            if log_bins[i] <= log_bins[i-1]:
                log_bins[i] = log_bins[i-1] + 1
    filter_matrix = torch.zeros(num_bands, num_bins, device=device)
    for i in range(num_bands):
        start_bin = log_bins[i]
        end_bin = log_bins[i+1] if i < num_bands - 1 else num_bins
        filter_matrix[i, start_bin:end_bin] = 1 / max(1, (end_bin - start_bin))

    return filter_matrix

def apply_log_filter(stft_output, filter_matrix):
    stft_output_transposed = stft_output.transpose(1, 2)
    filtered_output_transposed = torch.matmul(stft_output_transposed, filter_matrix.T)
    filtered_output = filtered_output_transposed.transpose(1, 2)
    return filtered_output