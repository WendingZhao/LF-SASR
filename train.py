import time
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from utils.dataloader import *
from model.SAnet_2 import Net
from einops import rearrange
# now the trian sample has 20 scenes
# test has 4 (HCI_NEW)

# parse the argument
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', type=bool, default=False)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--model_name', type=str, default='SAnet_2')

parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor")

parser.add_argument('--load_pretrain', type=bool, default=False)
parser.add_argument('--model_path', type=str, default='./log/mypth.tar')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patchsize_train', type=int, default=32, help='patchsize of LR images for training')

parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs to train')

parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
parser.add_argument('--trainset_dir', type=str, default='../autodl-tmp/Train_MDSR/')
parser.add_argument('--testset_dir', type=str, default='../autodl-tmp/Test_MDSR/')


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
        if idx_epoch % 50 == 0:
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




'''
对照部分

'''

from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from utils.MultiDegradation import LF_Blur, random_crop_SAI, LF_Bicubic, LF_Noise


def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    start_epoch = args.start_epoch
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)

    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f' % (idx_epoch + 1, loss_epoch_train))

        ''' Save PTH  '''
        if args.local_rank == 0:
            if args.task == 'SSR':
                save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
                    args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            elif args.task =='ASR':
                save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx%d_epoch_%02d_model.pth' % (
                    args.model_name, args.angRes_in, args.angRes_in, args.angRes_out, args.angRes_out, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))

        # ''' Validation '''
        step = 5
        if (idx_epoch + 1)%step==1 or idx_epoch > 60:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_list = []
                ssim_list = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr, ssim = test(args, test_name, test_loader, net, excel_file, save_dir)
                    excel_file.write_sheet(test_name, 'Average', 'PSNR', psnr)
                    excel_file.write_sheet(test_name, 'Average', 'SSIM', ssim)
                    excel_file.add_count(2)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                        idx_epoch + 1, test_name, psnr, ssim))
                    pass
                psnr_mean = np.array(psnr_list).mean()
                ssim_mean = np.array(ssim_list).mean()
                excel_file.write_sheet('ALL', 'Average', 'PSNR', psnr_mean)
                excel_file.write_sheet('ALL', 'Average', 'SSIM', ssim_mean)
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean, ssim_mean))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xlsx')
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        pass
    pass


def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    loss_list = []

    # set the degradation function
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig_min=args.sig_min, sig_max=args.sig_max,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
    )
    add_noise = LF_Noise(noise=args.noise, random=True)

    # LF 4 3 5 5 143 143
    # lf_defrad # LF 4 3 5 5 143 143
    # kernel 4 21 21
    # sigmas (4,)??
    for idx_iter, (LF) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        ''' degradation '''
        if args.task == 'SSR':
            # Isotropic or Anisotropic Gaussian Blurs
            [LF_degraded, kernels, sigmas] = blur_func(LF)
            LF, LF_degraded = random_crop_SAI(LF, LF_degraded, SAI_patch=args.patch_for_train*args.scale_factor) # the size changed
            # lf 128 128 degrad 32 32

            # down-sampling
            LF_degraded = LF_Bicubic(LF_degraded, scale=1/args.scale_factor)
            [LF_degraded, noise_levels] = add_noise(LF_degraded)
            # noise level (4,)

            LF_input = LF_degraded
            LF_target = LF
            # info is a list of tensors [(4 21 21) (4,) (4,) ]
            # the default value is all 0 except the sigmas has a value of 1e-6
            gt_blur = sigmas.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes_out, args.angRes_out) / 4

            gt_noise = noise_levels.repeat(1, 1, args.angRes_out, args.angRes_out)
            info = [kernels.to(device), gt_blur.to(device), gt_noise.to(device)]
        elif args.task == 'ASR':
            angFactor = (args.angRes_out - 1) // (args.angRes_in - 1)
            LF_sampled = LF[:, :, ::angFactor, ::angFactor, :, :]
            LF, LF_sampled = random_crop_SAI(LF, LF_sampled, SAI_patch=args.patch_for_train * args.scale_factor)

            LF_input = LF_sampled
            LF_target = LF
            info = None

        ''' super-resolve the degraded LF images'''
        LF_input = LF_input.to(device)      # low resolution
        LF_target = LF_target.to(device)    # high resolution
        net.train()

        LF_out = net(LF_input, info)
        loss = criterion(LF_out, LF_target, info)

        ''' calculate loss and PSNR/SSIM '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data.cpu())
        pass

    loss_mean = float(np.array(loss_list).mean())
    return loss_mean


def test(args, test_name, test_loader, net, excel_file, save_dir=None):
    psnr_list = []
    ssim_list = []

    # set the degradation function
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig=args.sig,
        lambda_1=args.lambda_1, lambda_2=args.lambda_2,
    )
    add_noise = LF_Noise(noise=args.noise, random=True)

    for idx_iter, (LF, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        ''' degradation '''
        if args.task == 'SSR':
            # Isotropic or Anisotropic Gaussian Blurs
            [LF_degraded, kernels, sigmas] = blur_func(LF)

            # down-sampling
            LF_degraded = LF_Bicubic(LF_degraded, scale=1/args.scale_factor)
            [LF_degraded, noise_levels] = add_noise(LF_degraded)

            LF_input = LF_degraded
            LF_target = LF
            # the info should change subsequently
            info = [kernels, sigmas, noise_levels]
        elif args.task == 'ASR':
            angFactor = (args.angRes_out - 1) // (args.angRes_in - 1)
            LF_sampled = LF[:, :, ::angFactor, ::angFactor, :, :]

            LF_input = LF_sampled
            LF_target = LF
            info = None

        ''' Crop LFs into Patches '''
        LF_divide_integrate_func = LF_divide_integrate(args.scale_factor, args.patch_size_for_test, args.stride_for_test)
        sub_LF_input = LF_divide_integrate_func.LFdivide(LF_input)
        # 64 3 5 5 32 32

        ''' SR the Patches '''
        sub_LF_out = []
        for i in range(0, sub_LF_input.size(0), args.minibatch_for_test):
            tmp = sub_LF_input[i:min(i + args.minibatch_for_test, sub_LF_input.size(0)), :, :, :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out = net(tmp.to(args.device), info)
                sub_LF_out.append(out['SR'])
        sub_LF_out = torch.cat(sub_LF_out, dim=0)
        LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
        LF_out = LF_out[:, :, :, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().detach()
        if LF_out.size(1)==1:
            LF_out = torch.cat([LF_out, LF_rgb2ycbcr(LF_target)[:, 1:3]], dim=1)
            LF_out = LF_ycbcr2rgb(LF_out)

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, LF_target, LF_out)
        excel_file.write_sheet(test_name, LF_name[0], 'PSNR', psnr)
        excel_file.write_sheet(test_name, LF_name[0], 'SSIM', ssim)
        excel_file.add_count(1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)

            # save the center view
            LF_out = (LF_out.squeeze(0).permute(1, 2, 3, 4, 0).cpu().detach().numpy().clip(0, 1) * 255).astype('uint8')
            path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
            img = LF_out[args.angRes_out//2, args.angRes_out//2, :, :, :]
            imageio.imwrite(path, img)


            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
                    img = LF_out[i, j, :, :, :]
                    imageio.imwrite(path, img)
                pass
        pass

    return [np.array(psnr_list).mean(), np.array(ssim_list).mean()]


def main_test(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 单/多gpu训练出现的module.?
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()
        psnr_list = []
        ssim_list = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            psnr, ssim = test(args, test_name, test_loader, net, excel_file, save_dir)
            excel_file.write_sheet(test_name, 'Average', 'PSNR', psnr)
            excel_file.write_sheet(test_name, 'Average', 'SSIM', ssim)
            excel_file.add_count(2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            print('Test on %s, psnr/ssim is %.2f/%.4f' % (test_name, psnr, ssim))
            pass

        psnr_mean = np.array(psnr_list).mean()
        ssim_mean = np.array(ssim_list).mean()
        excel_file.write_sheet('ALL', 'Average', 'PSNR', psnr_mean)
        excel_file.write_sheet('ALL', 'Average', 'SSIM', ssim_mean)
        print('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean, ssim_mean))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xlsx')

    pass


