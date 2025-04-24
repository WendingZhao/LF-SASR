import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.swin_transformer import SwinTransformer
import timm
from Distgblock import DistgBlock

class Net(nn.Module):
    def __init__(self, factor, angRes, code_dim=15):
        super().__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.factor = factor
        # PSF/blur 编码
        self.gen_code = Gen_Code(code_dim)
        # 初始卷积
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 深层级联卷积组
        self.deep_conv = CascadeGroups(n_group, n_block, angRes, channels, code_dim)
        # 上采样模块
        self.up_sample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 3, kernel_size=1, bias=False)
        )

    def forward(self, data):
        lf, blur, noise = data
        # 生成退化编码
        code = torch.cat((self.gen_code(blur), noise), dim=1)
        b, u, v, c, h, w = lf.shape
        # 初始特征
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer = self.initial_conv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        # 深层处理
        buffer = self.deep_conv(buffer, code)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        # 上采样
        out = self.up_sample(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        return out


class CascadeGroups(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels, code_dim):
        super().__init__()
        self.groups = nn.ModuleList([
            BasicGroup(n_block, angRes, channels, code_dim)
            for _ in range(n_group)
        ])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, code):
        identity = x
        for grp in self.groups:
            x = grp(x, code)
        b, u, v, c, h, w = x.shape
        buf = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buf)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        return out + identity


class BasicGroup(nn.Module):
    def __init__(self, n_block, angRes, channels, code_dim):
        super().__init__()
        # 使用 Swin Transformer 替代原 DABlock
        self.modulator = DABlock(channels, code_dim)
        # 多个 DistgBlock
        self.blocks = nn.ModuleList([
            DistgBlock(angRes, channels) for _ in range(n_block)
        ])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, code):
        identity = x
        x = self.modulator(x, code)  # 使用 swin 模块调制
        for blk in self.blocks:
            x = blk(x)
        b, u, v, c, h, w = x.shape
        buf = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buf)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        return out + identity

class SwinMLPBlock(nn.Module):
    def __init__(self, dim, code_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Sequential(
            nn.Linear(dim + code_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, code):
        B, N, C = x.shape
        code = code.unsqueeze(1).repeat(1, N, 1)  # (B, N, code_dim)
        x_res = x
        x = self.norm1(x)
        x = self.attn(torch.cat([x, code], dim=-1))
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x


class DABlock(nn.Module):
    def __init__(self, channels, code_dim=16):
        super().__init__()
        self.channels = channels
        self.patch_embed = nn.Conv2d(channels, channels, kernel_size=1)
        self.swin_block = SwinMLPBlock(channels, code_dim)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        x_2d = rearrange(x, 'b u v c h w -> (b u v) c h w')
        x_patch = self.patch_embed(x_2d)  # (B, C, H, W)

        H, W = h, w
        x_patch = x_patch.flatten(2).transpose(1, 2)  # (B, N, C)
        code = rearrange(code, 'b c u v -> (b u v) c')  # (B, code_dim)

        x_patch = self.swin_block(x_patch, code)
        x_patch = x_patch.transpose(1, 2).view(-1, self.channels, H, W)
        out = self.proj(x_patch)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x



class CA_Layer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channel_in, 16, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, channel_out, 1),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, code):
        b,u,v,c,h,w = x.shape
        fea = rearrange(x, 'b u v c h w -> (b u v) c h w')
        code_fea = self.avg_pool(fea)
        code_deg = rearrange(code, 'b c u v -> (b u v) c 1 1')
        cat = torch.cat((code_fea, code_deg), dim=1)
        att = self.mlp(cat).repeat(1,1,h,w)
        out = fea * att
        return rearrange(out, '(b u v) c h w -> b u v c h w', b=b,u=u,v=v)


class Gen_Code(nn.Module):
    def __init__(self, channel_out):
        super().__init__()
        kernel_size = 21
        ax = torch.arange(kernel_size).float() - kernel_size//2
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
        self.xx_yy = -(xx**2 + yy**2)
        self.gen = nn.Sequential(
            nn.Conv2d(kernel_size**2, 64, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, channel_out, 1, bias=False)
        )

    def forward(self, sigma):
        b, _, u, v = sigma.shape
        kernel = torch.exp(self.xx_yy.to(sigma.device) / (2 * sigma.view(-1,1,1)**2))
        kernel = kernel / kernel.sum([1,2], keepdim=True)
        kern = rearrange(kernel, '(b u v) h w -> b (h w) u v', b=b,u=u,v=v)
        return self.gen(kern)

if __name__ == "__main__":
    angRes = 5
    factor = 4

    batch_size=4
    net = Net(factor, angRes)
    # print(net)
    from thop import profile

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    input_lf = torch.randn(batch_size, angRes, angRes, 3, 32, 32).to(device)
    blur = torch.randn(batch_size, 1, angRes, angRes).to(device)
    noise = torch.randn(batch_size, 1, angRes, angRes).to(device)

    # THE PARAMATER OF THE MODEL
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=((input_lf, blur, noise), ))

    print('   Number of parameters: %.2fM' % (params / 1e6)) #3.90M origin
    print('   Number of FLOPs: %.2fG' % (flops / 1e9)) # 263.10G origin


