import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dim_mult = 2, num_groups = 16):
        # num_channels must be divisible by num_groups
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim_mult*in_channels, kernel_size, padding="same")
        self.norm1 = nn.GroupNorm(num_groups, dim_mult*in_channels)
        self.conv2 = nn.Conv2d(dim_mult*in_channels, out_channels, kernel_size, padding="same")
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.ReLU()

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)) # point wise convolution to match out_channels

    def forward(self, x):
        """
        x: torch.tensor of shape (batch_size, channels, frequency, time)
        """
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        return out + self.skip_connection(x) #element wise addition for residual link

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        # num_channels must be divisible by num_groups
        super().__init__()
        self.firstConv = nn.Conv2d(in_channels, 16, (3,3), padding="same")
        self.block1 = Block(16, 64, kernel_size=(3,3), dim_mult=2, num_groups=8)
        self.block2 = Block(64, 128, kernel_size=(3,3), dim_mult=2, num_groups=32)
        self.block3 = Block(128, 256, kernel_size=(3,3), dim_mult=2, num_groups=32)
        self.downSample = nn.MaxPool2d((2,2))

    def forward(self, x):
        print("x:", x.shape)
        x = self.firstConv(x)
        print("first Layer:", x.shape)
        x = self.block1.forward(x)
        print("block1:", x.shape)
        x = self.downSample(x)
        print("block1 downsampled:", x.shape)
        x = self.block2.forward(x)
        print("block2:", x.shape)
        x = self.downSample(x)
        print("block2 downsampled:", x.shape)
        x = self.block3.forward(x)
        print("block3:", x.shape)
        x = self.downSample(x)
        print("block3 downsampled:", x.shape)
        return x

class UpBlock():
    def __init__(self, out_channels):
        self.upSample1 = nn.ConvTranspose2d(256, 256, (4,4), stride = (2,2), padding=(1,1))
        self.block1 = Block(256, 128, kernel_size=(3,3), dim_mult=2, num_groups=32)
        self.upSample2 = nn.ConvTranspose2d(128, 128, (4,4), stride = (2,2), padding=(1,1))
        self.block2 = Block(128, 64, kernel_size=(3,3), dim_mult=2, num_groups=32)
        self.upSample3 = nn.ConvTranspose2d(64, 64, (4,4), stride = (2,2), padding=(1,1))
        self.block3 = Block(64, 16, kernel_size=(3,3), dim_mult=2, num_groups=8)
        self.lastConv = nn.Conv2d(16, out_channels, (3,3), padding="same")

    def forward(self, x):
        print("x:", x.shape)
        x = self.upSample1(x)
        print("upSample1:", x.shape)
        x = self.block1.forward(x)
        print("block1:", x.shape)
        x = self.upSample2(x)
        print("upSample2:", x.shape)
        x = self.block2.forward(x)
        print("block2:", x.shape)
        x = self.upSample3(x)
        print("upSample3:", x.shape)
        x = self.block3.forward(x)
        print("block3:", x.shape)
        x = self.lastConv(x)
        print("last Layer:", x.shape)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.downPath = DownBlock(in_channels)
        self.mid = Block(256, 256, (3,3), dim_mult=1, num_groups=32)
        self.upPath = UpBlock(out_channels)

    def forward(self, x):
        x = self.downPath(x)
        x = self.mid(x)
        x = self.upPath(x)