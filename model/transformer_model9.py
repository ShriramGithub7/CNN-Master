import torch
import torch.nn as nn

class UltimusBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UltimusBlock, self).__init__()
        self.K = nn.Linear(in_channels, out_channels)
        self.Q = nn.Linear(in_channels, out_channels)
        self.V = nn.Linear(in_channels, out_channels)
        self.Out = nn.Linear(out_channels, in_channels)
        
    def forward(self, x):
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        
        AM = nn.functional.softmax(torch.mm(Q, K.t()) / (8 ** 0.5), dim=1)
        Z = torch.mm(AM, V)
        out = self.Out(Z)
        
        return out

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ultimus1 = UltimusBlock(48, 8)
        self.ultimus2 = UltimusBlock(48, 8)
        self.ultimus3 = UltimusBlock(48, 8)
        self.ultimus4 = UltimusBlock(48, 8)
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc(x)
        return x
