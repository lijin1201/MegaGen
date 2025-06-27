import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = ConvBlock(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = ConvBlock(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = ConvBlock(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = ConvBlock(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d6 = self.up6(b)
        d6 = torch.cat([d6, e4], dim=1)
        d6 = self.dec6(d6)

        d7 = self.up7(d6)
        d7 = torch.cat([d7, e3], dim=1)
        d7 = self.dec7(d7)

        d8 = self.up8(d7)
        d8 = torch.cat([d8, e2], dim=1)
        d8 = self.dec8(d8)

        d9 = self.up9(d8)
        d9 = torch.cat([d9, e1], dim=1)
        d9 = self.dec9(d9)

        # Output
        return torch.sigmoid(self.final_conv(d9))  # For binary segmentation

# Example usage:
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    print(model)
    x = torch.randn((1, 3, 256, 256))  # (batch_size, channels, height, width)
    y = model(x)
    print("Output shape:", y.shape)
