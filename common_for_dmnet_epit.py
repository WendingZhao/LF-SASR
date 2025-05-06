from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from utils.MultiDegradation import LF_Blur, random_crop_SAI, LF_Bicubic, LF_Noise
from model.SAnet_epit_test import Net


def test_for_dmnet_epit(args, test_name, test_loader, net, excel_file, save_dir=None):
    psnr_list = []
    ssim_list = []

    # set the degradation function
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig=args.sig,
        lambda_1=args.lambda_1, lambda_2=args.lambda_2,
    )
    add_noise = LF_Noise(noise=args.noise_test, random=False)

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
            gt_blur = sigmas.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, args.angRes_out, args.angRes_out) / 4

            gt_noise = noise_levels.repeat(1, 1, args.angRes_out, args.angRes_out)
            info = [kernels.to(args.device), gt_blur.to(args.device), gt_noise.to(args.device)]

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
                out = net((tmp.to(args.device), gt_blur.repeat(tmp.shape[0], 1, 1, 1).to(args.device), gt_noise.repeat(tmp.shape[0], 1, 1, 1).to(args.device)))

                sub_LF_out.append(out['SR'])
        sub_LF_out = torch.cat(sub_LF_out, dim=0)
        LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
        LF_out = LF_out[:, :, :, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().detach()
        if LF_out.size(1)==1:
            LF_out = torch.cat([LF_out, LF_rgb2ycbcr(LF_target)[:, 1:3]], dim=1)
            LF_out = LF_ycbcr2rgb(LF_out)

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, LF_target, LF_out)
        # excel_file.write_sheet(test_name, LF_name[0], 'PSNR', psnr)
        # excel_file.write_sheet(test_name, LF_name[0], 'SSIM', ssim)
        # excel_file.add_count(1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        ''' Save RGB '''
        if save_dir is not None:
            pass # temporary banned for this part

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

def main_test_dm_epit(args):
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
    net=Net(args.scale_factor,args.angRes_out)
    net.to(args.device)
    cudnn.benchmark = True
    ''' Load Pre-Trained PTH '''
    ckpt_path = args.path_pre_pth
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model = torch.load(args.model_path, map_location={'cuda:0': args.device})
    net.load_state_dict(model['state_dict'], strict=False)
    # 单/多gpu训练出现的module.?
    # try:
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['state_dict'].items():
    #         name = 'module.' + k  # add `module.`
    #         new_state_dict[name] = v
    #     # load params
    #     net.load_state_dict(new_state_dict)
    #     print('Use pretrain model!')
    # except:
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['state_dict'].items():
    #         new_state_dict[k] = v
    #     # load params
    #     net.load_state_dict(new_state_dict)
    #     print('Use pretrain model!')
    #     pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    for noise in [0, 15]: # defult=[0, 15, 50]
        args.noise_test = noise
        for sig in [0, 1.5]: # default=[0, 1.5, 3]
            args.sig = sig
            test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_list = []
                ssim_list = []
                for index, test_name in enumerate(test_Names):
                    torch.cuda.empty_cache()
                    test_loader = test_Loaders[index]

                    save_dir = result_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)
                    psnr, ssim = test_for_dmnet_epit(args, test_name, test_loader, net, excel_file, save_dir)

                    # excel_file.write_sheet(test_name, 'Average', 'PSNR', psnr)
                    # excel_file.write_sheet(test_name, 'Average', 'SSIM', ssim)
                    # excel_file.add_count(2)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    print('Dataset--%15s,\t noise--%f, \t sig---%f, \t PSNR--%f, \t SSIM---%f' % (
                        test_name, args.noise_test, sig, psnr, ssim))
                    pass
                psnr_mean = np.array(psnr_list).mean()
                ssim_mean = np.array(ssim_list).mean()
                # excel_file.write_sheet('ALL', 'Average', 'PSNR', psnr_mean)
                # excel_file.write_sheet('ALL', 'Average', 'SSIM', ssim_mean)
                print('The mean psnr on testsets is %.5f, mean ssim is %.5f(noise--%f,sig---%f)' % (psnr_mean, ssim_mean,args.noise_test,args.sig))
                # logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f(noise--%f,sig---%f)' % (psnr_mean, ssim_mean,args.noise_test,args.sig))
                # excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xlsx')
                pass
            pass


    pass