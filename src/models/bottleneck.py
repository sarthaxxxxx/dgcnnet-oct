import torch 
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, 
                out_channels, 
                stride = 1, 
                dilation = 1, 
                downsample = None, 
                multi_grid = 1
                ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = stride,
                              padding = dilation * multi_grid,
                              dilation = dilation * multi_grid,
                              bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace = False)
        self.relu_inplace = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride 
        self.dilation = dilation

    def forward(self, a):
        res = a
        a = self.conv1(a)
        a = self.bn1(a)
        a = self.relu(a)

        a = self.conv2(a)
        a = self.bn2(a)
        a = self.relu(a)

        a = self.conv3(a)
        a = self.bn3(a)

        if self.downsample is not None:
            a = self.downsample(a)
        
        return self.relu_inplace(a + res) 