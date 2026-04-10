import torch
import torch.nn as nn
from saicinpainting.training.modules.pix2pixhd import NLayerDiscriminator, get_norm_layer
from saicinpainting.training.modules.base import BaseDiscriminator

class MultiScalePatchGAN(BaseDiscriminator):
    """
    Multi-Scale PatchGAN discriminator.
    Uses NLayerDiscriminator at 3 different scales: 1x, 0.5x, 0.25x.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer='bn', num_D=3):
        super().__init__()
        self.num_D = num_D
        norm_layer = get_norm_layer(norm_layer)
        
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        result = []
        all_features = []
        input_downsampled = x
        
        for i in range(self.num_D):
            pred, features = self.discriminators[i](input_downsampled)
            result.append(pred)
            all_features.append(features)
            if i < self.num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        
        # Return the final prediction (list of predictions) and list of feature maps
        # This structure matches how common multi-scale GAN trainers expect it
        return result, all_features
