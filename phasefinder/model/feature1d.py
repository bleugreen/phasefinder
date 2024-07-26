import torch.nn as nn

class FeatureExtraction(nn.Module):
    def __init__(self, num_bands=81, num_channels=20):
        super(FeatureExtraction, self).__init__()

        self.conv1 = nn.Conv1d(num_bands, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1) 
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1) 
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)  
        x = self.elu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)  
        x = self.elu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)  
        x = self.elu3(x)
        x = self.dropout3(x)

        return x
