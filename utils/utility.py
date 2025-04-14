import torch
import numpy as np
from skimage import metrics
import torch.nn.functional as F
from einops import rearrange


def save_ckpt(args, net, idx_epoch):
    if args.parallel:
        torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                   './log/' + args.model_name + '_' + str(args.upfactor) + 'xSR.tar')
    else:
        torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                   './log/' + args.model_name + '_' + str(args.upfactor) + 'xSR.tar')

    if idx_epoch % 50 == 0:
        if args.parallel:
            torch.save({'epoch': idx_epoch, 'state_dict': net.module.state_dict()},
                       './log_my/' + args.model_name + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')
        else:
            torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict()},
                       './log_my/' + args.model_name + '_' + str(args.upfactor) + 'xSR' + '_epoch_' + str(idx_epoch) + '.tar')


def cal_metrics(label, out):

    U, V, C, H, W = label.size()

    # cal the ycbcr
    label_y = (65.481 * label[:, :, 0, :, :] + 128.553 * label[:, :, 1, :, :] + 24.966 * label[:, :, 2, :, :] + 16) / 255.0
    out_y = (65.481 * out[:, :, 0, :, :] + 128.553 * out[:, :, 1, :, :] + 24.966 * out[:, :, 2, :, :] + 16) / 255.0

    # the output should be limited to 0 to 1
    label_y = label_y.data.cpu().numpy().clip(0, 1)
    out_y = out_y.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            PSNR[u, v] = metrics.peak_signal_noise_ratio(label_y[u, v, :, :], out_y[u, v, :, :])
            SSIM[u, v] = metrics.structural_similarity(label_y[u, v, :, :], out_y[u, v, :, :], data_range=1 , gaussian_weights=True)
    # the ssim should use data range
    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean


def ImageExtend(Im, bdr):
    """
    对图像进行镜像扩展，用于处理边缘区域，使得在后续处理（如卷积）时不会丢失边缘信息。

    参数：
    Im: 输入图像张量，形状为 (N, C, H, W)。
    bdr: 边界扩展参数，包含四个元素 [bdr_top, bdr_bottom, bdr_left, bdr_right]，
         表示上下左右四个方向的扩展量。

    返回：
    扩展后的图像张量，裁剪后的形状为 (N, C, H + bdr_top + bdr_bottom, W + bdr_left + bdr_right)。
    """


    [_, _, h, w] = Im.size()

    # left right up down diagonal
    Im_lr = torch.flip(Im, dims=[-1])      # 左右翻转
    Im_ud = torch.flip(Im, dims=[-2])      # 上下翻转
    Im_diag = torch.flip(Im, dims=[-1, -2])# 对角线翻转（先左右再上下）

    # dim=-1 宽度方向拼接
    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)

    # 高度方向拼接
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)

    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    # 图像是左上到右下的坐标系

    return Im_out


def LFdivide(lf, patch_size, stride):
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF


def LFintegrate(subLFs, patch_size, stride):
    """
    将分割后的光场图像块重新拼接成完整图像。

    参数：
    subLFs: 分割后的子块张量，形状为 (N1, N2, U, V, C, patch_size, patch_size)。
    patch_size: 子块的大小。
    stride: 分割时的步长。

    返回：
    拼接后的完整光场图像，形状为 (U, V, C, H, W)。
    """
    n1, n2, u, v, c, h, w = subLFs.shape
    bdr = (patch_size - stride) // 2
    # 提取有效区域（去除扩展的边界）
    # 取每个块的中间stride×stride区域（去掉边缘bdr）
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    # 重新排列维度：
    # 将 (n1, n2, U, V, C, stride, stride) 转换为
    # (U, V, C, n1*stride, n2*stride)
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> u v c (n1 h) (n2 w)')

    return outLF


def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0
    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]

    return y
