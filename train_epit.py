from common import main
from option import args

args.task = 'SSR'
args.max_angRes = 5
args.patch_for_train = 32
args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.data_list = ['HCI_new', 'HCI_old', 'Stanford_Gantry']

    args.scale_factor = 4
    args.angRes_in = 5
    args.angRes_out = 5

    'part code is not finished'
    args.model_name = 'EPIT_0'
    args.start_epoch = 0
    main(args)
