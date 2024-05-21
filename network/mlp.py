import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, 
                in_channels,
                hid_channels,
                out_channels):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.softmax(out)

        return out
