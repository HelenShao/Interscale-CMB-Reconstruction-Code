from __future__ import annotations
import pickle
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=negative_slope, inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

    def forward(self, x):
        return self.model(x)

class UNET(nn.Module):

    def __init__(self, in_channels, out_channels, feature_dims=None, negative_slope=0.01, mean_init=None):
        super().__init__()
        if feature_dims is None:
            feature_dims = [32, 64, 128, 256, 512, 1024]
        self.downsamplings = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.transposes = nn.ModuleList()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(feature_dims[-1], feature_dims[-1] * 2, negative_slope)
        self.final = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)
        self.mean_init = mean_init
        for feature_dim in feature_dims:
            self.downsamplings.append(DoubleConv(in_channels, feature_dim, negative_slope))
            in_channels = feature_dim
        for feature_dim in feature_dims[::-1]:
            self.transposes.append(nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=2, stride=2))
            self.upsamplings.append(DoubleConv(feature_dim * 2, feature_dim, negative_slope))
        if mean_init is not None:
            with open(mean_init, 'rb') as f:
                loaded_stats_dict = pickle.load(f)
            train_y_mean = loaded_stats_dict['train_y_mean']
            nn.init.zeros_(self.final.weight)
            nn.init.constant_(self.final.bias, train_y_mean)

    def forward(self, x):
        skip_connections = []
        for downsampling in self.downsamplings:
            x = downsampling(x)
            skip_connections.append(x)
            x = self.pooling(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for (idx, (transpose, upsampling)) in enumerate(zip(self.transposes, self.upsamplings)):
            x = transpose(x)
            if x.shape != skip_connections[idx].shape:
                x = TF.resize(x, size=skip_connections[idx].shape[2:])
            x = torch.concat((skip_connections[idx], x), dim=1)
            x = upsampling(x)
        return self.final(x)
