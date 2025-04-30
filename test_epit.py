from common import main_test as main
from common import main_test_dm as train
import argparse
import torch.backends.cudnn as cudnn
from utils.utility import *
from utils.dataloader import *
from model.SAnet_0 import Net
from numpy import random

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

    args.scale_factor = 4
    args.model_name = 'EPIT'
    args.path_pre_pth = './pth/SAnet_epit_4xSR_epoch_1000.tar'
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
    args.path_pre_pth = './log/DAnet_tmp_800.tar'
    torch.multiprocessing.set_start_method('spawn')


    train(args)


