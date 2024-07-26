import torch.nn as nn

class BeatPhaseDecoder(nn.Module):
    def __init__(self, num_tcn_outputs, num_classes):
        super(BeatPhaseDecoder, self).__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(num_tcn_outputs, 72)
        self.relu = nn.ELU()
        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(72, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x
