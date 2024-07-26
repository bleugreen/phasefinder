import torch
import numpy as np

def calculate_beat_phase(num_frames, beat_times, sr, hop, K=360):
    frame_rate = sr / hop  # Frames per second
    beat_indices = torch.round(torch.tensor(beat_times) * frame_rate).long()
    phase = torch.zeros(num_frames)
    # Add num_frames as a virtual beat end point to include the last segment
    beat_indices = torch.tensor(sorted(beat_indices), dtype=torch.long)

    for i in range(len(beat_indices) -1):
        start_idx = beat_indices[i]
        end_idx = beat_indices[i + 1]
        num_intervals = end_idx - start_idx

        # Fill the phase up to just before the next beat index
        if num_intervals > 0:  
            phase_step = torch.linspace(0, K, num_intervals + 1)[:-1]
            end = min(num_frames, end_idx)
            phase_end = end-start_idx
            if end > start_idx:
                phase[start_idx:end] = phase_step[:phase_end]

    return phase.int()

def triangular_label(width):
    peak = (width + 1) / 2 
    tri_list= [min(i, width - i + 1) / peak for i in range(1, width + 1)]
    return torch.tensor(tri_list) / np.sum(tri_list)

def generate_blurred_one_hots_wrapped(indices, K=360, width=5):
    blur_vector = triangular_label(width)
    num_frames = indices.shape[-1]

    one_hots = torch.zeros((num_frames, K))

    for offset, weight in enumerate(blur_vector, -len(blur_vector)//2):
        wrapped_indices = (indices + offset) % K
        one_hots[torch.arange(num_frames), wrapped_indices] += weight

    one_hots /= one_hots.sum(dim=1, keepdim=True)

    return one_hots.to_sparse()
