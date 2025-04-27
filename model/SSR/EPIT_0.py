import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        self.angRes = args.angRes_in
        self.scale = args.scale_factor
        self.gen_code = Gen_Code(15)

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        # self.altblock = nn.Sequential(
        #     AltFilter(self.angRes, channels),
        #     AltFilter(self.angRes, channels),
        #     AltFilter(self.angRes, channels),
        #     AltFilter(self.angRes, channels),
        #     AltFilter(self.angRes, channels),
        # )

        Groups = []
        for i in range(5):
            Groups.append(BasicGroup(self.angRes,channels))
        # 这个指针指向了四个BasicGroup模块的列表
        self.Group = nn.Sequential(*Groups)

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        [b, c, u, v, h, w] = lr.size()
        # 4 3 5 5 32

        # rgb2ycbcr
        # 32 32 128 128
        lr_ycbcr = LF_rgb2ycbcr(lr)
        sr_ycbcr = LF_interpolate(lr_ycbcr, scale_factor=self.scale, mode='bicubic')

        # Initial Feature Extraction
        # x 4 1 25 32 32
        x = rearrange(lr_ycbcr[:, 0:1, :, :, :, :], 'b c u v h w -> b c (u v) h w')

        # 4 64 25 32 32
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer # res connect
        # 4 64 25 32 32


        [kernels, sigmas, noise_levels]=info
        code = torch.cat((self.gen_code(sigmas), noise_levels), dim=1)
        tmp1=buffer
        # Deep Spatial-Angular Correlation Learning
        for i in range(5):
            buffer = self.Group[i](buffer, code)
        # buffer = self.altblock(buffer,code) + buffer # res connect +alt learning
        buffer=buffer+tmp1
        # UP-Sampling
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=u, v=v)
        y = self.upsampling(buffer)
        y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        # ycbcr2rgb
        sr_ycbcr[:, 0:1, :, :, :, :] += y
        out = {}
        out['SR'] = LF_ycbcr2rgb(sr_ycbcr)
        return out

class BasicGroup(nn.Module):
    def __init__(self, angRes, channels):
        super(BasicGroup, self).__init__()
        # 退化感知模块，负责根据 code 调整特征
        self.DAB = DABlock(channels)

        self.altblock=AltFilter(angRes, channels)

    def forward(self, x, code):
        tmp=x
        x=rearrange(x,'b c (u v) h w -> b u v c h w',u=5,v=5)
        b, u, v, c, h, w = x.shape

        buffer = self.DAB(x, code)

        buffer=rearrange(buffer,'b u v c h w -> b c (u v) h w')
        buffer=self.altblock(buffer)

        return buffer+tmp # res module


# transformer可用于提取epi序列的长依赖
class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(BasicTrans, self).__init__()
        # 特征映射
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, dropout, bias=False)
        # initalize the weight of attention through kaiming init
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))

        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(spa_dim*2, spa_dim, bias=False),
            nn.Dropout(dropout)
        )

        # 特征反映射
        # linear_out: Linear(in_features=128, out_features=64, bias=False)
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, k_h: int, k_w: int):
        # 根据局部感受野生成注意力mask
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h // 2
        k_h_right = k_h - k_h_left
        k_w_left = k_w // 2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()
        attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)

        # 展平为 token 序列
        epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()

        self.angRes = angRes

        self.epi_trans = BasicTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        # 4 64 25 32 32
        shortcut = buffer
        # res connect
        [_, _, _, h, w] = buffer.size()

        # 设置注意力窗口大小
        self.epi_trans.mask_field = [self.angRes * 2, 11] #list [10,11]

        # Horizontal
        # 4 64 160 5 32
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)

        # 4 64 25 32 32
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        return buffer


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass

class DABlock(nn.Module):
    def __init__(self, channels):
        super(DABlock, self).__init__()

        #
        self.generate_kernel = nn.Sequential(
            nn.Conv2d(16, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64 * 9, 1, 1, 0, bias=False))

        self.conv_1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.ca_layer = CA_Layer(80, 64)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, code_array):
        b, u, v, c, h, w = x.shape
        # 将6D光场输入展平为标准的2D图像形式，以便进行group卷积
        input_spa = rearrange(x, 'b u v c h w -> 1 (b u v c) h w', b=b, u=u, v=v)

        # 使用退化代码生成每个位置的卷积核： (b, 64*9, u, v) -> 卷积核 shape: (b*u*v, 64*9)
        kernel = self.generate_kernel(code_array)  # b, 64 * 9, u, v
        kernel = rearrange(kernel, 'b c u v -> (b u v) c')

        # 使用自定义卷积核进行group-wise卷积，每个位置使用不同核
        fea_spa = self.relu(F.conv2d(
            input_spa,
            kernel.contiguous().view(-1, 1, 3, 3),  # 每组一个3x3卷积核
            groups=b * u * v * c,
            padding=1
        ))

        # 还原为标准格式 (b u v) c h w
        fea_spa = rearrange(fea_spa, '1 (b u v c) h w -> (b u v) c h w', b=b, u=u, v=v, c=c)
        # 特征通道映射
        fea_spa_da = self.conv_1x1(fea_spa)
        # reshape to standard format
        fea_spa_da = rearrange(fea_spa_da, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        # 加上通道注意力输出和残差连接
        out = fea_spa_da + self.ca_layer(fea_spa_da, code_array) + x

        return out


class CA_Layer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(CA_Layer, self).__init__()
        # 多层感知机（MLP）用于生成注意力权重，输出维度为 channel_out
        self.mlp = nn.Sequential(
            nn.Conv2d(channel_in, 16, 1, 1, 0),       # 降维到 16 维
            nn.LeakyReLU(0.1, True),                  # 激活函数
            nn.Conv2d(16, channel_out, 1, 1, 0),       # 再升维到 channel_out
            nn.Sigmoid())                             # 将权重压缩到 0~1 之间

        # 自适应全局平均池化，将特征图转换为每个通道的均值（全局特征）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        fea = rearrange(x, 'b u v c h w -> (b u v) c h w')


        # 对每张 feature map 做通道级平均池化（提取图像本身语义信息）
        code_fea = self.avg_pool(fea)                      # [b*u*v, c, 1, 1]

        # 把每个视角对应的退化编码 code 也 reshape 成一样的格式
        code_deg = rearrange(code, 'b c u v -> (b u v) c 1 1')  # [b*u*v, c, 1, 1]

        # 拼接图像信息和退化信息（沿 channel 方向拼接）
        code = torch.cat((code_fea, code_deg), dim=1)      # [b*u*v, c*2, 1, 1]

        # 通过 MLP 得到 attention 权重
        att = self.mlp(code)                               # [b*u*v, channel_out, 1, 1]

        # 将注意力权重广播到特征图尺寸
        att = att.repeat(1, 1, h, w)                        # [b*u*v, c, h, w]

        # 对原始特征图进行通道加权（即注意力机制）
        out = fea * att                                     # [b*u*v, c, h, w]

        # reshape 回原始光场格式
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class Gen_Code(nn.Module):
    def __init__(self, channel_out):
        super(Gen_Code, self).__init__()
        kernel_size = 21
        # 构造一维坐标，中心为0，范围[-10,10]
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        # 构造横向坐标网格，尺寸为 1 x kernel_size x kernel_size
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
        # 构造纵向坐标网格，尺寸为 1 x kernel_size x kernel_size，使用repeat_interleave保证行内相同数值连续出现
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
        # 根据高斯公式计算权重指数部分的负平方距离：- (xx^2 + yy^2)
        self.xx_yy = -(xx ** 2 + yy ** 2)

        # 构建一个由连续1x1卷积和LeakyReLU激活构成的序列模块

        '''
        是否也可以将测得的psf展平作为输入
        '''

        self.gen_code = nn.Sequential(
            nn.Conv2d(kernel_size ** 2, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, sigma):
        # 获取输入 sigma 的形状信息：批大小b，通道数c（可能为1），以及u和v维度
        [b, c, u, v] =sigma.size()
        # 将预先计算的 -(xx^2 + yy^2) 扩散到与输入 sigma 同样的设备上
        # 计算高斯核，注意 sigma: 对每个样本都要计算一个对应的高斯核
        # 公式：exp( -(xx^2+yy^2) / (2 * sigma^2) )，
        # 这里 sigma.view(-1, 1, 1) 将 sigma 展平，以适应每个样本计算
        kernel = torch.exp(self.xx_yy.to(sigma.device) / (2. * sigma.view(-1, 1, 1) ** 2)).to(sigma.device)
        # kernel 100 21 21? xxyy 12121

        # 对高斯核进行归一化，使其所有元素和为1，归一化维度为[1,2]（即核尺寸维度）
        kernel = kernel / kernel.sum([1, 2], keepdim=True)

        # 将高斯核的维度重排以适应 gen_code 中的卷积输入
        # 输入shape 从 (b*u*v, h, w) 重排为 (b, h*w, u, v)
        # 说明：这里 kernel 原本的批量维度为 (b * u * v)，重组后将 h*w 作为新通道数
        kernel = rearrange(kernel, '(b u v) h w -> b (h w) u v', b=b, u=u, v=v)
        code = self.gen_code(kernel)

        return code


if __name__ == "__main__":
    angRes = 5
    factor = 4

    batch_size=4
    from option import args

    net = get_model(args)
    # print(net)
    from thop import profile

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(net)
    lf=torch.randn(batch_size, 3,angRes, angRes,  32, 32).to(device)

    kernels=torch.randn(batch_size, 21,21).to(device)
    sigmas=torch.randn(batch_size, 1, angRes, angRes).to(device)
    noise_levels=torch.randn(batch_size, 1, angRes, angRes).to(device)

    info=[kernels, sigmas, noise_levels]

    # THE PARAMATER OF THE MODEL
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(lf,info))

    print('   Total number of parameters: %.2fM' % (total / 1e6))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

'''
   the origin
   Number of parameters: 1.14M
   Number of FLOPs: 217.22G
   
   the modified
    Number of parameters: 1.41M
   Number of FLOPs: 219.37G
'''