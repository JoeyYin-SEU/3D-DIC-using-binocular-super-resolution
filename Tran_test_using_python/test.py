from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--using_epipolar', type=bool, default=False)
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--overlapping_test', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='iPASSR_2xSR')
    return parser.parse_args()


def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    if cfg.overlapping_test:
        if cfg.using_epipolar:
            model = torch.load('./log/non_ep/' + cfg.model_name + '.pth.tar')
        else:
            model = torch.load('./log/using_ep/' + cfg.model_name + '.pth.tar')
    else:
        if cfg.using_epipolar:
            model = torch.load('./log/using_ep/' + cfg.model_name + '.pth.tar')
        else:
            model = torch.load('./log/non_ep/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])

    if cfg.using_epipolar:
        file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/using_ep/lr')
    else:
        file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/non_ep/lr')
    for idx in range(len(file_list)):

        if cfg.using_epipolar:
            LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/using_ep/lr/' + file_list[idx] + '/lr0.png')
            LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/using_ep/lr/' + file_list[idx] + '/lr1.png')
        else:
            LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/non_ep/lr/' + file_list[idx] + '/lr0.png')
            LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/non_ep/lr/' + file_list[idx] + '/lr1.png')
        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
        scene_name = file_list[idx]
        print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            SR_left, SR_right = net(LR_left, LR_right, is_training=0)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
        if cfg.overlapping_test:
            if cfg.using_epipolar:
                save_path = './results/u_in_nmodel/' + cfg.model_name + '/' + cfg.dataset
            else:
                save_path = './results/n_in_umodel/' + cfg.model_name + '/' + cfg.dataset
        else:
            if cfg.using_epipolar:
                save_path = './results/using_ep/' + cfg.model_name + '/' + cfg.dataset
            else:
                save_path = './results/non_ep/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name + '_L.png')
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    cfg.using_epipolar=False
    cfg.overlapping_test=False
    cfg.dataset = 'testing_img'
    test(cfg)
    print('Finished_1!')
    cfg = parse_args()
    cfg.using_epipolar=True
    cfg.overlapping_test=False
    cfg.dataset = 'testing_img'
    test(cfg)
    print('Finished_2!')
    cfg = parse_args()
    cfg.using_epipolar=False
    cfg.overlapping_test=True
    cfg.dataset = 'testing_img'
    test(cfg)
    print('Finished_3!')
    cfg = parse_args()
    cfg.using_epipolar=True
    cfg.overlapping_test=True
    cfg.dataset = 'testing_img'
    test(cfg)
    print('Finished!')
