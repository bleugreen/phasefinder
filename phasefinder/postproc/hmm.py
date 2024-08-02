import torch

def hmm_beat_estimation(phase_prediction, bpm, frame_rate, bpm_confidence=0.9, distance_threshold_factor=0.2, device='cpu'):
    num_states = phase_prediction.shape[1]
    seq_len = phase_prediction.shape[0]
    transition_probs = calculate_transition_probs(num_states, bpm, frame_rate, bpm_confidence, distance_threshold_factor, device)
    emission_probs = phase_prediction.to(device)
    viterbi_probs = torch.zeros((seq_len, num_states), device=device)
    backpointers = torch.zeros((seq_len, num_states), dtype=torch.long, device=device)
    viterbi_probs[0] = emission_probs[0] + transition_probs[0]
    for t in range(1, seq_len):
        prev_probs = viterbi_probs[t-1].unsqueeze(1)
        curr_emis = emission_probs[t].unsqueeze(0)
        curr_probs = prev_probs + transition_probs + curr_emis
        viterbi_probs[t], backpointers[t] = torch.max(curr_probs, dim=0)
    beat_positions = backtrack(backpointers)
    return beat_positions

def calculate_transition_probs(num_states, bpm, frame_rate, bpm_confidence, distance_threshold_factor, device='cpu'):
    frames_per_beat = frame_rate * 60 / bpm
    phase_change_per_frame = 360 / frames_per_beat
    i = torch.arange(num_states, device=device).float() * (360 / num_states)
    j = torch.arange(num_states, device=device).float() * (360 / num_states)
    expected_phase_diff = (i.unsqueeze(1) + phase_change_per_frame) % 360 - j.unsqueeze(0)
    expected_phase_diff = torch.min(abs(expected_phase_diff), 360 - abs(expected_phase_diff))
    
    distance_threshold = phase_change_per_frame * distance_threshold_factor
    transition_probs = torch.where(expected_phase_diff <= distance_threshold, 1.0 - (expected_phase_diff / distance_threshold), torch.tensor(1e-10, device=device))
    
    uniform_probs = torch.ones_like(transition_probs) / num_states
    transition_probs = bpm_confidence * transition_probs + (1 - bpm_confidence) * uniform_probs
    
    transition_probs = transition_probs / transition_probs.sum(dim=1, keepdim=True)
    return torch.log(transition_probs)

def backtrack(backpointers):
    seq_len = backpointers.shape[0]
    beat_positions = []
    curr_state = torch.argmax(backpointers[-1])
    beat_positions.append(curr_state.item())
    for t in range(seq_len - 2, -1, -1):
        curr_state = backpointers[t, curr_state]
        beat_positions.append(curr_state.item())
    beat_positions = beat_positions[::-1]
    return beat_positions