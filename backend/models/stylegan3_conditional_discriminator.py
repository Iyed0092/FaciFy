# backend/models/stylegan3_conditional_discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_resolution=512, img_channels=3, num_classes=10, base_channels=64):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes

        self.label_embed = nn.Embedding(num_classes, img_resolution * img_resolution)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0)
        )

    def forward(self, x, labels):
        label_map = self.label_embed(labels).view(-1, 1, self.img_resolution, self.img_resolution)
        x = torch.cat([x, label_map], dim=1)
        return self.net(x).view(-1)

if __name__ == "__main__":
    model = ConditionalDiscriminator(img_resolution=64, num_classes=5)
    dummy_img = torch.randn(2, 3, 64, 64)
    dummy_labels = torch.tensor([1, 3])
    out = model(dummy_img, dummy_labels)
    print("Output shape:", out.shape)
