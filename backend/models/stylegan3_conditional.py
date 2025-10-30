# backend/models/stylegan3_conditional.py
import torch
import torch.nn as nn
from stylegan3.torch_utils.ops import filtered_lrelu
from stylegan3.torch_utils import persistence

class ConditionalMappingNetwork(nn.Module):
    def __init__(self, z_dim=512, c_dim=10, w_dim=512, num_layers=8):
        super().__init__()
        self.num_layers = num_layers
        self.fc = nn.ModuleList()
        for i in range(num_layers):
            in_dim = z_dim + c_dim if i == 0 else w_dim
            self.fc.append(nn.Linear(in_dim, w_dim))
            self.fc.append(nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        for layer in self.fc:
            x = layer(x)
        return x

class ConditionalGenerator(nn.Module):
    def __init__(self, w_dim=512, img_channels=3, img_resolution=512):
        super().__init__()
        self.w_dim = w_dim
        self.img_channels = img_channels
        self.img_resolution = img_resolution

        self.fc = nn.Linear(w_dim, 4*4*512)
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Conv2d(128, img_channels, 3, 1, 1)
        ])

    def forward(self, w):
        x = self.fc(w).view(-1, 512, 4, 4)
        for conv in self.conv_blocks[:-1]:
            x = filtered_lrelu(conv(x))
        x = torch.tanh(self.conv_blocks[-1](x))
        return x

if __name__ == "__main__":
    z = torch.randn(2, 512)
    c = torch.zeros(2, 10)
    mapping = ConditionalMappingNetwork()
    w = mapping(z, c)
    gen = ConditionalGenerator()
    img = gen(w)
    print("Generated image shape:", img.shape)
