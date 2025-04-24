import torch
import torch.nn as nn
from einops import rearrange


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
        x_patch = x_patch.transpose(1, 2).view(-1, c, H, W)
        out = self.proj(x_patch)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


if __name__ == "__main__":
    angRes = 5
    factor = 4
    batch_size = 2
    C = 64
    H = W = 32

    dablock = DABlock(channels=C, code_dim=16)
    dablock.eval()

    lf = torch.randn(batch_size, angRes, angRes, C, H, W)
    code = torch.randn(batch_size, 16, angRes, angRes)

    with torch.no_grad():
        out = dablock(lf, code)
        print('Input shape:', lf.shape)
        print('Output shape:', out.shape)
        print('Pass complete.')
