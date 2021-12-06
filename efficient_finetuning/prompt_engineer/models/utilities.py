import torch
import torch.nn as nn

class MaxPool1dFixed(nn.Module):
    "Pytorch maxpool's behavior expects a third dimension when doing a 2D maxpool, hence this implementation"
    def __init__(self, kernel):
        super().__init__()
        self.max_pool_layer= nn.MaxPool1d(kernel)
    
    def forward(self, x):
        return torch.squeeze(self.max_pool_layer(x.unsqueeze(0)))


class AvgPool1dFixed(nn.Module):
    "Pytorch avgpool's behavior expects a third dimension when doing a 2D maxpool, hence this implementation"
    def __init__(self, kernel):
        super().__init__()
        self.avg_pool_layer= nn.AvgPool1d(kernel)
    
    def forward(self, x):
        return torch.squeeze(self.avg_pool_layer(x.unsqueeze(0)))