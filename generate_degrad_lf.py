import argparse
import os
import h5py
import numpy as np
from pathlib import Path
import scipy.io as scio
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from utils.MultiDegradation import LF_Blur, random_crop_SAI, LF_Bicubic, LF_Noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='SSR', help='SSR, ASR')
    parser.add_argument("--max_angRes", type=int, default=5, help="angular resolution")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='../datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='../Data_for_dest/', help='')
    return parser.parse_args()


def main(args):
    angRes = args.max_angRes

    ''' save dir '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath(str(args.task) + '_' + str(angRes) + 'x' + str(angRes))
    save_dir.mkdir(exist_ok=True)

    ''' generating .h5 date from .mat files '''
    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig=args.sig,
        lambda_1=args.lambda_1, lambda_2=args.lambda_2,
    )
    add_noise = LF_Noise(noise=args.noise_test, random=False)
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['HCI_new', 'HCI_old', 'Stanford_Gantry']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF = np.array(data['LF'])

                (U, V, H, W, _) = LF.shape
                H = H // 4 * 4
                W = W // 4 * 4

                # Extract central angRes * angRes views
                LF = LF[4, 4, 0:H, 0:W, 0:3]
                LF = LF.astype('double')

                idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                HR_LF = np.zeros((H, W, 3), dtype='single')
                HR_LF[:] = LF

                [LF_degraded, kernels, sigmas] = blur_func(LF)

                # down-sampling
                LF_degraded = LF_Bicubic(LF_degraded, scale=1/args.scale_factor)
                [LF_degraded, noise_levels] = add_noise(LF_degraded)


                # save
                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']



                print('%d training samples have been generated\n' % (idx_scene_save))
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)