import torch

import torch.nn as nn

class UNet(nn.Module):
    """
    UNet architecture for image segmentation.
    Input: (batch_size, 3, 64, 64)
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 64)   # (N, 64, 64, 64)
        self.pool1 = nn.MaxPool2d(2)                   # (N, 64, 32, 32)
        self.enc2 = self.conv_block(64, 128)           # (N, 128, 32, 32)
        self.pool2 = nn.MaxPool2d(2)                   # (N, 128, 16, 16)
        self.enc3 = self.conv_block(128, 256)          # (N, 256, 16, 16)
        self.pool3 = nn.MaxPool2d(2)                   # (N, 256, 8, 8)
        self.enc4 = self.conv_block(256, 512)          # (N, 512, 8, 8)
        self.pool4 = nn.MaxPool2d(2)                   # (N, 512, 4, 4)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)   # (N, 1024, 4, 4)

        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # (N, 512, 8, 8)
        self.dec4 = self.conv_block(1024, 512)         # (N, 512, 8, 8)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # (N, 256, 16, 16)
        self.dec3 = self.conv_block(512, 256)          # (N, 256, 16, 16)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)   # (N, 128, 32, 32)
        self.dec2 = self.conv_block(256, 128)          # (N, 128, 32, 32)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)    # (N, 64, 64, 64)
        self.dec1 = self.conv_block(128, 64)           # (N, 64, 64, 64)

        # Output layer
        self.conv_last = nn.Conv2d(64, out_channels, 1) # (N, out_channels, 64, 64)

    def conv_block(self, in_channels, out_channels):
        """
        Two convolutional layers with ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        print(f"Input: {x.shape}")  # (N, 3, 64, 64)
        enc1 = self.enc1(x)
        print(f"After enc1: {enc1.shape}")  # (N, 64, 64, 64)
        p1 = self.pool1(enc1)
        print(f"After pool1: {p1.shape}")   # (N, 64, 32, 32)

        enc2 = self.enc2(p1)
        print(f"After enc2: {enc2.shape}")  # (N, 128, 32, 32)
        p2 = self.pool2(enc2)
        print(f"After pool2: {p2.shape}")   # (N, 128, 16, 16)

        enc3 = self.enc3(p2)
        print(f"After enc3: {enc3.shape}")  # (N, 256, 16, 16)
        p3 = self.pool3(enc3)
        print(f"After pool3: {p3.shape}")   # (N, 256, 8, 8)

        enc4 = self.enc4(p3)
        print(f"After enc4: {enc4.shape}")  # (N, 512, 8, 8)
        p4 = self.pool4(enc4)
        print(f"After pool4: {p4.shape}")   # (N, 512, 4, 4)

        bottleneck = self.bottleneck(p4)
        print(f"After bottleneck: {bottleneck.shape}")  # (N, 1024, 4, 4)

        up4 = self.upconv4(bottleneck)
        print(f"After upconv4: {up4.shape}")  # (N, 512, 8, 8)
        cat4 = torch.cat([up4, enc4], dim=1)
        print(f"After cat4: {cat4.shape}")    # (N, 1024, 8, 8)
        dec4 = self.dec4(cat4)
        print(f"After dec4: {dec4.shape}")    # (N, 512, 8, 8)

        up3 = self.upconv3(dec4)
        print(f"After upconv3: {up3.shape}")  # (N, 256, 16, 16)
        cat3 = torch.cat([up3, enc3], dim=1)
        print(f"After cat3: {cat3.shape}")    # (N, 512, 16, 16)
        dec3 = self.dec3(cat3)
        print(f"After dec3: {dec3.shape}")    # (N, 256, 16, 16)

        up2 = self.upconv2(dec3)
        print(f"After upconv2: {up2.shape}")  # (N, 128, 32, 32)
        cat2 = torch.cat([up2, enc2], dim=1)
        print(f"After cat2: {cat2.shape}")    # (N, 256, 32, 32)
        dec2 = self.dec2(cat2)
        print(f"After dec2: {dec2.shape}")    # (N, 128, 32, 32)

        up1 = self.upconv1(dec2)
        print(f"After upconv1: {up1.shape}")  # (N, 64, 64, 64)
        cat1 = torch.cat([up1, enc1], dim=1)
        print(f"After cat1: {cat1.shape}")    # (N, 128, 64, 64)
        dec1 = self.dec1(cat1)
        print(f"After dec1: {dec1.shape}")    # (N, 64, 64, 64)

        out = self.conv_last(dec1)
        print(f"Output: {out.shape}")         # (N, out_channels, 64, 64)
        return out

# Example usage:
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)

import matplotlib.pyplot as plt

# Visualize input and output
input_img = x[0].detach().cpu().numpy().transpose(1, 2, 0)
output_img = out[0, 0].detach().cpu().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("UNet Output")
plt.imshow(output_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()