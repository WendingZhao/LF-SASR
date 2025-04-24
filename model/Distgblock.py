import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super().__init__()
        self.spa_conv = SpaConv(channels, channels)
        self.ang_conv = AngConv(angRes, channels, channels//4)
        self.epi_conv = EpiConv(angRes, channels, channels//2)
        self.fuse = nn.Sequential(
            nn.Conv2d(2*channels + channels//4, channels, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        b,u,v,c,h,w = x.shape
        spa = self.spa_conv(x)
        ang = self.ang_conv(x)
        epih = self.epi_conv(x)
        xT = rearrange(x, 'b u v c h w -> b v u c w h')
        epiv = rearrange(self.epi_conv(xT), 'b v u c w h -> b u v c h w', b=b,u=u,v=v)
        fea = torch.cat((spa, ang, epih, epiv), dim=3)
        fea = rearrange(fea, 'b u v c h w -> (b u v) c h w')
        out = self.fuse(fea)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b,u=u,v=v)
        return out + x


class SpaConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, channel_out, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        b,u,v,c,h,w = x.shape
        inp = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(inp)
        return rearrange(out, '(b u v) c h w -> b u v c h w', b=b,u=u,v=v)


class AngConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes*angRes*channel_out, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(angRes)
        )

    def forward(self, x):
        b,u,v,c,h,w = x.shape
        inp = rearrange(x, 'b u v c h w -> (b h w) c u v')
        out = self.conv(inp)
        return rearrange(out, '(b h w) c u v -> b u v c h w', b=b,h=h,w=w)


class EpiConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, padding=(0,angRes//2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes*channel_out, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            PixelShuffle1D(angRes)
        )

    def forward(self, x):
        b,u,v,c,h,w = x.shape
        inp = rearrange(x, 'b u v c h w -> (b u h) c v w')
        out = self.conv(inp)
        return rearrange(out, '(b u h) c v w -> b u v c h w', b=b,u=u,h=h)


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        return x.view(b, c, h*self.factor, w)