from common import main_test as main
from common import main_test_dm as train
from common import main_test_epit_0 as test_epit_0
import argparse
import torch.backends.cudnn as cudnn
from utils.utility import *
from utils.dataloader import *
from model.SAnet_0 import Net
from numpy import random
from common_for_dmnet_epit import main_test_dm_epit

from option import args
args.task = 'SSR'
args.max_angRes = 5
args.batch_size = 4
args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    # args.data_list = ['HCI_new']
    args.scale_factor = 4
    args.model_name = 'EPIT_0'
    args.path_pre_pth = './pth/EPIT_0_5x5_4x_epoch_64_model.pth'
    for index in range(5, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        # test_epit_0(args)
        main(args)
if __name__ == '__main__suspend':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/EPIT_5x5_4x_model.pth'
    for index in range(5, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        test_epit_0(args)

if __name__ == '__main__suspend':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/EPIT_multi_5x5_4x.pth'
    for index in range(5, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        test_epit_0(args)

if __name__ == '__main__suspend':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT_0'
    args.path_pre_pth = './pth/EPIT_0_5x5_4x_epoch_64_model.pth'
    for index in range(5, args.max_angRes + 1):
        args.angRes_in = index
        args.angRes_out = index
        main(args)

    # args.scale_factor = 2
    # args.model_name = 'EPIT'
    # args.path_pre_pth = './pth/EPIT_5x5_2x_model.pth'
    # for index in range(1, args.max_angRes + 1):
    #     args.angRes_in = index
    #     args.angRes_out = index
    #     main(args)


'''
dmnet test
'''
if __name__ == '__main__suspend':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './log/LF-DAnet_4xSR_epoch_950.tar'
    torch.multiprocessing.set_start_method('spawn')


    train(args)


if __name__ == '__main__SUSPEND':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/SAnet_epit_4xSR_epoch_1000.tar'
    torch.multiprocessing.set_start_method('spawn')


    main_test_dm_epit(args)