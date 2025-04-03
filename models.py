import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DNet(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        
        # First 1D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=18, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(18)
        
        # Second 1D convolutional layer
        self.conv2 = nn.Conv1d(in_channels=18, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
    
        # Fully connected layer for regression output
        self.fc1 = nn.Linear(8 * 5, 32) # 5 = 45 / 3 / 3 (2 rounds of max pooling)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = F.max_pool1d(torch.relu(self.bn1(self.conv1(x))), 3)
        out = F.max_pool1d(torch.relu(self.bn2(self.conv2(out))), 3)
        # Flatten the output for the fully connected layer
        out = out.view(out.size(0), -1)
        
        # Pass through the fully connected layer
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
 