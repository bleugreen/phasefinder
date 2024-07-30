from torch.utils.data import Dataset
import h5py
import torch
from utils.one_hots import generate_blurred_one_hots_wrapped


class BeatDataset(Dataset):
    def __init__(self, data_path, group, mode='both', items=None, device='cuda', phase_width=5):
        """
        Initializes the dataset.
        :param data_path: Path to the HDF5 file.
        :param group: The group within the HDF5 file ('train', 'val', 'test').
        :param mode: 'beat', 'downbeat', or 'both' (default: 'both').
        :param items: List of items to include in the dataset (default: None).
                      Options: 'stft', 'phase', 'label', 'time', 'filepath', 'bpm'.
        :param device: Device to use for tensors (default: 'cuda').
        """
        self.data_path = data_path
        self.group = group
        self.mode = mode
        self.phase_width = phase_width
        self.items = items if items is not None else ['stft', 'phase', 'label', 'time', 'filepath', 'bpm']
        self.device = device

        with h5py.File(self.data_path, 'r') as file:
            self.keys = list(file[group].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as file:
            data = file[self.group][self.keys[idx]]
            result = []

            for item in self.items:
                if item == 'stft':
                    spec = torch.from_numpy(data['stft'][...]).to(self.device)
                    result.append(spec)
                elif item == 'phase':
                    if self.mode == 'beat' or self.mode == 'both':
                        beat_phase = torch.from_numpy(data['beat_phase'][...]).long()
                        beat_phase = generate_blurred_one_hots_wrapped(beat_phase, width=self.phase_width).to_dense().unsqueeze(0).to(self.device)
                        result.append(beat_phase)
                    if self.mode == 'downbeat' or self.mode == 'both':
                        downbeat_phase = torch.from_numpy(data['downbeat_phase'][...]).long()
                        downbeat_phase = generate_blurred_one_hots_wrapped(downbeat_phase, width=self.phase_width).to_dense().unsqueeze(0).to(self.device)
                        result.append(downbeat_phase)
                elif item == 'label':
                    if self.mode == 'beat' or self.mode == 'both':
                        beat_phase = torch.from_numpy(data['beat_phase'][...]).long().to(self.device)
                        result.append(beat_phase)
                    if self.mode == 'downbeat' or self.mode == 'both':
                        downbeat_phase = torch.from_numpy(data['downbeat_phase'][...]).long().to(self.device)
                        result.append(downbeat_phase)
                elif item == 'time':
                    if self.mode == 'beat' or self.mode == 'both':
                        beats = torch.from_numpy(data.attrs['beats'][...]).float()
                        result.append(beats)
                    if self.mode == 'downbeat' or self.mode == 'both':
                        downbeats = torch.from_numpy(data.attrs['downbeats'][...]).float()
                        result.append(downbeats)
                elif item == 'filepath':
                    filepath = data.attrs['filepath']
                    result.append(filepath)
                elif item == 'bpm':
                    bpm = data.attrs['bpm']
                    result.append(bpm)

            return tuple(result)
