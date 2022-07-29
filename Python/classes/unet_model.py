""" Full assembly of the parts to form the complete network """

# import torch.nn.functional as F
from classes.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.up4 = Up(32, 16 * factor, bilinear)
        self.outc = OutConv(16, n_classes)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    print('check up')
    # net = UNet(1,1)
    # input = torch.randn(1, 1, 64, 1000).float()
    # print(input.size())
    #
    # output = net(input)
    # print(output.size())
    #
    # target = torch.randn(64*1000).float()  # a dummy target, for example
    # target = target.view(1, 1, 64, 1000)  # make it the same shape as output
    # criterion = nn.MSELoss()
    #
    # loss = criterion(output, target)
    # print(loss)
    #
    # net.zero_grad()     # zeroes the gradient buffers of all parameters
    #
    # loss.backward()
