import torch.nn as nn
from pytorch_tcn import TCN

from phasefinder.model.feature1d import FeatureExtraction
from phasefinder.model.decoder import BeatPhaseDecoder
from phasefinder.model.attention import AttentionModule


class PhasefinderModelAttn(nn.Module):
    def __init__(self, num_bands=81, num_channels=36, num_classes=360, 
                 kernel_size=5, dropout=0.1, num_tcn_layers=16, dilation=8):
        super(PhasefinderModelAttn, self).__init__()
        self.feature_extraction = FeatureExtraction(num_bands=num_bands, num_channels=num_channels)
        self.tcn_beat = TCN(
            num_inputs=num_channels,
            num_channels=[num_channels] * num_tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=False,
            use_skip_connections=True,
            kernel_initializer='kaiming_normal',
            use_norm='layer_norm',
            activation='relu',
            dilation_reset=dilation
        )
        self.attention = AttentionModule(num_channels)
        self.decoder_beat = BeatPhaseDecoder(num_tcn_outputs=num_channels, num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extraction(x)
        features_flat = features.view(features.size(0), features.size(1), -1)
        tcn_output_beat = self.tcn_beat(features_flat).transpose(1, 2)

        tcn_output_beat = self.attention(tcn_output_beat)

        beat_phase_output = self.decoder_beat(tcn_output_beat.unsqueeze(2))
        return beat_phase_output
