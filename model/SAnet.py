import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from model.Distgblock import DistgBlock

class Net(nn.Module):
    def __init__(self, factor, angRes):
        super(Net, self).__init__()
        channels = 64
        # 定义级联分组模块的组数与每组的块数
        n_group = 4
        n_block = 4
        self.factor = factor
        # gencode's input is the number of channel out
        self.gen_code = Gen_Code(15)
        # 初始卷积层：将输入的RGB图像（3通道）映射到64个特征通道
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 深层级联卷积组：使用 CascadeGroups 模块，
        # 参数包括分组数n_group、每组中的块数n_block、角度分辨率angRes（具体用于配置卷积操作），以及通道数channels
        self.deep_conv = CascadeGroups(n_group, n_block, angRes, channels)

        self.up_sample = nn.Sequential(
            # 1. 使用1x1卷积将channels扩展到 channels * factor**2，
            # 为后续PixelShuffle重排做准备（使通道数变为上采样因子对应的平方倍
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            # 2. 使用LeakyReLU激活函数，参数0.1指定负半轴斜率，加速非线性映射
            nn.LeakyReLU(0.1, True),

            # 3. PixelShuffle：利用上一步骤扩大后的通道信息，重排为高分辨率特征图
            # PixelShuffle会把多余的通道重整为空间维度，实现上采样
            nn.PixelShuffle(factor),

            # 4. 最后再用1x1卷积将通道数映射回3，生成输出RGB图像
            nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, data):
        (lf, blur, noise) = data
        # 级联
        code = torch.cat((self.gen_code(blur), noise), dim=1)
        b, u, v, c, h, w = lf.shape

        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer = self.initial_conv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        # deep conv 四个模块
        buffer = self.deep_conv(buffer, code)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')

        # upsample
        out = self.up_sample(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class CascadeGroups(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeGroups, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(BasicGroup(n_block, angRes, channels))
        # 这个指针指向了四个BasicGroup模块的列表
        self.Group = nn.Sequential(*Groups)

        # 3*3 conv，等同于init conv
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        buffer = x
        # 依次通过各个 BasicGroup 进行特征处理，
        # 每个组都会接收当前的 buffer 和生成的条件编码 code
        for i in range(self.n_group):
            buffer = self.Group[i](buffer, code)

        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')

        out = self.conv(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x # 残差模块


class BasicGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(BasicGroup, self).__init__()
        # 退化感知模块，负责根据 code 调整特征
        self.DAB = DABlock(channels)

        # n个distg block
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))

        # 合并block
        self.block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape

        buffer = self.DAB(x, code)

        for i in range(self.n_block):
            buffer = self.block[i](buffer)

        # 打平视角维度，方便 2D 卷积处理
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')

        out = self.conv(buffer)

        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x # res module

# this part in the article is DM-block not DAblock
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
        # kernel 400 576
        # 使用自定义卷积核进行group-wise卷积，每个位置使用不同核
        fea_spa = self.relu(F.conv2d(
            input_spa,
            kernel.contiguous().view(-1, 1, 3, 3),  # 每组一个3x3卷积核 reshape 后 25600 1 3 3
            groups=b * u * v * c,
            padding=1
        ))

        # 还原为标准格式 (b u v) c h w 400 64 32 32
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
        b, c, u, v = sigma.shape
        # 将预先计算的 -(xx^2 + yy^2) 扩散到与输入 sigma 同样的设备上
        # 计算高斯核，注意 sigma: 对每个样本都要计算一个对应的高斯核
        # 公式：exp( -(xx^2+yy^2) / (2 * sigma^2) )，
        # 这里 sigma.view(-1, 1, 1) 将 sigma 展平，以适应每个样本计算
        kernel = torch.exp(self.xx_yy.to(sigma.device) / (2. * sigma.view(-1, 1, 1) ** 2))

        # 对高斯核进行归一化，使其所有元素和为1，归一化维度为[1,2]（即核尺寸维度）
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        # kernel 400 21 21
        # 将高斯核的维度重排以适应 gen_code 中的卷积输入
        # 输入shape 从 (b*u*v, h, w) 重排为 (b, h*w, u, v)
        # 说明：这里 kernel 原本的批量维度为 (b * u * v)，重组后将 h*w 作为新通道数
        kernel = rearrange(kernel, '(b u v) h w -> b (h w) u v', b=b, u=u, v=v)
        # after rearrange 16 441 5 5
        code = self.gen_code(kernel)
        # (16,15 ,5 ,5)
        return code


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



    print('   Total number of parameters: %.2fM' % (total / 1e6))

    print('   Number of parameters: %.2fM' % (params / 1e6)) #3.90M origin
    print('   Number of FLOPs: %.2fG' % (flops / 1e9)) # 263.10G origin

'''
   Number of parameters: 3.80M
   Number of FLOPs: 263.10G
   
      Total number of parameters: 3.80M
   Number of parameters: 3.80M
   Number of FLOPs: 263.10G
'''

