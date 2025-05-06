import time
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from utils.dataloader import *
from model.SAnet_epit import Net
from einops import rearrange


# parse the argument
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', type=bool, default=False)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--model_name', type=str, default='SAnet_epit')

parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")

parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--model_path', type=str, default='./log/mypth.tar')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--patchsize_train', type=int, default=32, help='patchsize of LR images for training')

parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')

parser.add_argument('--n_steps', type=int, default=300, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")

# parser.add_argument('--trainset_dir', type=str, default='../autodl-tmp/Train_MDSR/')
# parser.add_argument('--testset_dir', type=str, default='../autodl-tmp/Test_MDSR/')

parser.add_argument('--trainset_dir', type=str, default='../Data/Train_MDSR/')
parser.add_argument('--testset_dir', type=str, default='../Data/Test_MDSR/')


# cooperative training setting
parser.add_argument('--coop_mode', type=bool ,default=False)
parser.add_argument('--coop_trainset_dir', type=str, default='../autodl-tmp/Train_MDSR/')
parser.add_argument('--coop_testset_dir', type=str, default='../autodl-tmp/Test_MDSR/')





args = parser.parse_args()


def train(args):
    net = Net(factor=args.upfactor, angRes=args.angRes)
    net.to(args.device)

    # benchamark 加速模式
    cudnn.benchmark = True
    epoch_state = 0

    if args.load_pretrain:
        if os.path.isfile(args.model_path):
            model = torch.load(args.model_path, map_location={'cuda:0': args.device})
            net.load_state_dict(model['state_dict'], strict=False)
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(args.load_model))

    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0])

    # 创建日志目录和文件
    log_dir = './log_transfomer_2/'
    os.makedirs(log_dir, exist_ok=True)
    loss_log_file = os.path.join(log_dir, f"{args.model_name}_loss_log.txt")
    loss_evaluate_file= os.path.join(log_dir, f"{args.model_name}_loss_evaluate.txt")
    # 初始化损失日志文件（添加标题）
    if not os.path.exists(loss_log_file):
        with open(loss_log_file, 'w') as f:
            f.write("Epoch, Loss\n")
    if not os.path.exists(loss_evaluate_file):
        with open(loss_evaluate_file, 'w') as f:
            f.write("init!\n")

    # 配置损失函数和优化器
    criterion_Loss = torch.nn.L1Loss().to(args.device)
    # 仅优化需要梯度的参数
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=args.lr)
    # 学习率衰减策略，每n个epoch衰减一次，衰减因子为gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []

    # 主训练循环
    for idx_epoch in range(epoch_state, args.n_epochs):
        # 清理缓存
        torch.cuda.empty_cache()

        # 初始化多级降质模块 用于生成lr
        gen_LR = MultiDegrade(
            scale=args.upfactor,
            kernel_size=21,
            sig_min=0,
            sig_max=4,
        )

        # initializa the training set and dataloader

        train_set = TrainSetLoader(args)
        train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=True # 每个epoch打乱数据
                                  )

        # 训练循环
        for idx_iter, lf_hr in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 调整张量维度，合并视角维度
            lfimg_hr = rearrange(lf_hr, 'b u v c h w -> b (u v) c h w')
            # 生成lr及降质参数
            lfimg_lr, sigma, noise_level = gen_LR(lfimg_hr.to(args.device))
            # 恢复维度
            lf_lr = rearrange(lfimg_lr, 'b (u v) c h w -> b u v c h w', u=args.angRes, v=args.angRes)

            # 计算裁剪边界，去除边缘模糊区域
            bdr = 12 // args.upfactor
            # 裁剪高分辨率标签和输入低分辨率数据
            label = lf_hr[:, :, :, :, 12:-12, 12:-12]  # 裁剪HR标签
            data = lf_lr[:, :, :, :, bdr:-bdr, bdr:-bdr]  # 裁剪LR输入

            # 数据增强（随机翻转/旋转等）
            label, data = augmentation(label, data)

            # 扩展降质参数到视角维度
            # /4是归一化
            gt_blur = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes, args.angRes) / 4
            gt_noise = noise_level.repeat(1, 1, args.angRes, args.angRes)

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播（输入包含LR数据和降质参数）
            out_sr = net((data, gt_blur, gt_noise))
            # 计算损失并反向传播
            loss = criterion_Loss(out_sr, label.to(args.device))
            loss.backward()
            optimizer.step()
            # loss.detach()新版用法，在此不改变
            loss_epoch.append(loss.data.cpu())  # 记录当前batch损失
        # average loss
        avg_loss=float(np.array(loss_epoch).mean())
        print(time.ctime()[4:-5] + ' Epoch----%5d, loss_sr---%f' % (idx_epoch + 1, avg_loss))

        with open(loss_log_file, 'a') as f:
            f.write(f"{idx_epoch + 1}, {avg_loss:.6f}\n")

        # 导向utility
        save_ckpt(args, net, idx_epoch + 1)

        ''' evaluation '''
        if idx_epoch % 50 == 0 and idx_epoch>=800:
            for noise in [0, 15]: # defult=[0, 15, 50]
                args.noise = noise
                for sig in [0, 1.5]: # default=[0, 1.5, 3]
                    args.sig = sig
                    # 导向utils的dataloader
                    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
                    for index, test_name in enumerate(test_Names):
                        torch.cuda.empty_cache()
                        test_loader = test_Loaders[index]
                        # valid
                        psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                        print('Dataset--%15s,\t noise--%f, \t sig---%f, \t PSNR--%f, \t SSIM---%f' % (
                        test_name, args.noise, sig, psnr_epoch_test, ssim_epoch_test))
                        txtfile = open(loss_evaluate_file, 'a')
                        txtfile.write('Epoch--%f,\t Dataset--%15s,\t noise--%f,\t sig--%f,\t PSNR---%f,\t SSIM---%f \n' % (
                            idx_epoch + 1, test_name, args.noise, sig, psnr_epoch_test, ssim_epoch_test))
                        txtfile.close()

            txtfile = open(loss_evaluate_file, 'a')
            txtfile.write('\n')
            txtfile.close()
        # 学习率调度器更新
        scheduler.step()


def coop_train(args):

    net = Net(factor=args.upfactor, angRes=args.angRes)
    ...

    ...

def valid(test_loader, net):
    psnr_iter_test = []  # 存储每个测试批次的PSNR值
    ssim_iter_test = []  # 存储每个测试批次的SSIM值

    for idx_iter, (data, label, sigma, noise_level) in enumerate(test_loader):
        # 获取测试数据的降质参数（模糊和噪声）
        # 将模糊参数扩展到视角维度，并归一化
        gt_blur = sigma.unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes, args.angRes) / 4
        # 将噪声参数扩展到视角维度
        gt_noise = noise_level.repeat(1, 1, args.angRes, args.angRes)

        # 根据是否启用裁剪模式选择不同处理方式
        if args.crop == False:
            # 直接推理（无需分块处理）
            with torch.no_grad():
                # 前向传播获取超分辨率结果
                outLF = net(data.to(args.device))  # 将数据移动到设备（如GPU）
                outLF = outLF.squeeze()  # 去除多余的维度（如batch维度为1时）
        else:
            # 分块推理模式（处理大尺寸图像防止内存溢出）
            patch_size = args.patchsize_test  # 分块尺寸
            data = data.squeeze()  # 移除单例维度（如batch维度）

            # 将输入光场图像分割为小块（分块重叠）
            sub_lfs = LFdivide(data, patch_size, patch_size // 2)  # 分割函数

            # 调整分块张量形状以便批量处理
            n1, n2, u, v, c, h, w = sub_lfs.shape  # 分块维度信息
            sub_lfs = rearrange(  # 将(n1, n2, u, v, c, h, w) → (n1*n2, u, v, c, h, w)
                sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w'
            )

            # 分批次处理分块（防止显存不足）
            mini_batch = args.minibatch_test  # 每个批次的分块数量
            num_inference = (n1 * n2) // mini_batch  # 总批次数

            with torch.no_grad():
                out_lfs = []  # 存储所有分块的输出

                # 遍历每个批次进行推理
                for idx_inference in range(num_inference):
                    torch.cuda.empty_cache()  # 释放显存
                    # 取当前批次的分块
                    input_lfs = sub_lfs[  # 从分块列表中取出mini_batch数量的块
                                idx_inference * mini_batch : (idx_inference+1) * mini_batch,
                                :, :, :, :, :
                                ]

                    # 前向传播（输入包含分块数据和扩展的降质参数）
                    lf_out = net(
                        (
                            input_lfs.to(args.device),  # 分块数据移动到设备
                            # 扩展模糊参数到当前批次的样本数量
                            gt_blur.repeat(input_lfs.shape[0], 1, 1, 1),
                            # 扩展噪声参数到当前批次的样本数量
                            gt_noise.repeat(input_lfs.shape[0], 1, 1, 1)
                        )
                    )
                    out_lfs.append(lf_out)  # 存储当前批次的输出

                # 处理剩余不足一个批次的分块（如果有）
                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    start_idx = num_inference * mini_batch
                    input_lfs = sub_lfs[start_idx :, :, :, :, :, :]
                    lf_out = net(
                        (
                            input_lfs.to(args.device),
                            gt_blur.repeat(input_lfs.shape[0], 1, 1, 1),
                            gt_noise.repeat(input_lfs.shape[0], 1, 1, 1)
                        )
                    )
                    out_lfs.append(lf_out)

                # 合并所有批次的输出
                out_lfs = torch.cat(out_lfs, dim=0)  # 在batch维度拼接

                # 恢复分块的原始结构
                out_lfs = rearrange(  # 将(n1*n2, ...) → (n1, n2, ...)
                    out_lfs,
                    '(n1 n2) u v c h w -> n1 n2 u v c h w',
                    n1=n1, n2=n2
                )

                # 将分块结果重新组合为完整光场图像
                outLF = LFintegrate(  # 合并函数
                    out_lfs,
                    patch_size * args.upfactor,  # 分块放大后的尺寸
                    patch_size * args.upfactor // 2  # 分块重叠区域
                )

                # 调整输出尺寸到原始高分辨率标签的尺寸
                # 注意：假设data.shape[3]和data.shape[4]是原始低分辨率的高宽
                outLF = outLF[:, :, :,
                        0 : data.shape[3] * args.upfactor,  # 高（height）方向
                        0 : data.shape[4] * args.upfactor   # 宽（width）方向
                        ]

        # 计算当前批次的PSNR和SSIM
        psnr, ssim = cal_metrics(label.squeeze(), outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    # 计算整个测试集的平均PSNR和SSIM
    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        y = torch.flip(y, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        x = torch.flip(x, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x, y = x[:, :, :, color, :, :], y[:, :, :, color, :, :]

    return x, y


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    print(args.coop_mode)
    if args.coop_mode:

        coop_train(args)
    else:
        train(args)