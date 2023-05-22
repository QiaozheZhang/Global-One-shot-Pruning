import numpy as np
import torch 
from torchvision import datasets, transforms
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib, math
import tqdm, sys
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from utils import *
from trainer import *
#%matplotlib inline

# enable cuda devices
import os

def cal_gw(density_list, grids_list, R, ep, all_num, trace):
    temp_gw = 0
    max_ev_index = density_list.index(max(density_list))
    ev_threshold = grids_list[max_ev_index]

    for i in range(len(density_list)):
        density = density_list[i]
        eigenvalue = abs(grids_list[i])
        """ if eigenvalue <= 0:
            eigenvalue = 1e-30 """
        radius_2 = 2*ep/eigenvalue
        temp_gw += density*radius_2/(R**2+radius_2)
    
    gw = temp_gw/sum(density_list)
    print(gw)

    return gw

def get_params_diff(model, s_model):
    param_list = []
    s_param_list = []
    norm2 = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_list.append(param)

    for param in s_model.parameters():
        if not param.requires_grad:
            continue
        s_param_list.append(param)
    
    for i in range(len(param_list)):
        diff = param_list[i] - s_param_list[i]
        norm2 += torch.norm(diff, p=2)**2
    
    return norm2**0.5

def sparse_list(sparse_density_list, max_density_index, sparse_density):
    if max(sparse_density_list) > sparse_density:
        sparse_density_list[max_density_index] = max(sparse_density_list) - sparse_density
        print('big')
        done = True
    else:
        sparse_density_list[max_density_index] = 0
        sparse_density = sparse_density - max(sparse_density_list)
        max_density_index = sparse_density_list.index(max(sparse_density_list))
        done = False

    return sparse_density_list, sparse_density, done, max_density_index

def cal_sparse_rate(index, trace, important_ev_num=0):
    init_file_path()
    train_loader, val_loader, actual_val_loader = get_dataset()
    criterion = get_criterion()
    parser_args.mode = 'sparse'

    eigen_path = parser_args.file_path + '{}/hessien_eigenvalue/'.format(parser_args.mode)
    check_dir(eigen_path)

    model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, 0)
    model = torch.load(model_path, map_location='cuda:{}'.format(parser_args.gpu))
    all_num, norm2, max_w = get_params_size(model)
    model.eval()
    ori_loss, loss_std = cal_loss(model, train_loader, criterion)
    validate(model, val_loader, criterion)

    ss_model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, index)
    ss_model = torch.load(ss_model_path, map_location='cuda:{}'.format(parser_args.gpu))
    ss_model.eval()
    ss_ori_loss, ss_loss_std = cal_loss(ss_model, train_loader, criterion)

    loss_std = loss_std #ori_loss + loss_std - ss_ori_loss
    print(loss_std)

    ep = loss_std#.detach().cpu()
    temp_gw = 0
    R = norm2
    density_list = []
    density_f = open(parser_args.file_path + 'density.txt',"r")
    density_lines = density_f.readlines()
    for density_line in density_lines:
        density = float(density_line[:-1])
        density_list.append(density)
    density_f.close()
    grids_list = []
    grids_f = open(parser_args.file_path + 'grids.txt',"r")
    grids_lines = grids_f.readlines()
    for grids_line in grids_lines:
        grids = float(grids_line[:-1])
        grids_list.append(grids)
    grids_f.close()

    for i in range(len(grids_list)):
        d_index = len(grids_list) - i
        if density_list[d_index-1] == 0:
            stop_index = d_index - 1
            break

    csv_data = pd.read_csv(parser_args.file_path + 'sparse/acc_loss.csv')
    orign_acc = csv_data['acc'][0]
    for i in range(parser_args.sparse_epoch):
        sparse_acc = csv_data['acc'][i]
        if sparse_acc < orign_acc-0.1:
            sparse_threshold = i/parser_args.sparse_epoch
            print(sparse_threshold)
            break

    gaussian_zero_num_rate = important_ev_num
    print(gaussian_zero_num_rate)
    sparse_density_sum = gaussian_zero_num_rate/(parser_args.sparse_epoch-gaussian_zero_num_rate) * sum(density_list)
    new_density_list, new_grids_list = [], []
    for w in range(len(density_list)):
        if density_list[w] > 0:
            new_density_list.append(density_list[w])
            new_grids_list.append(grids_list[w])
    density_list, grids_list = new_density_list, new_grids_list
    density_list.append(sparse_density_sum)
    grids_list.append(1e-30)

    print('calculate gaussian width')
    gw_list = []
    for i in range(int(norm2)+1):
        R = i
        gw = cal_gw(density_list, grids_list, R, ep, all_num, trace)
        gw_list.append(gw)


    model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, index)
    model = torch.load(model_path, map_location='cuda:{}'.format(parser_args.gpu))
    print('calculate R & saprse rate')
    R_list, sparse_rate_list, acc_list, sparse_rate_list1, R_list1 = [], [], [], [], []
    for i in range(parser_args.sparse_epoch):
        print(i)
        s_model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, i)
        s_model = torch.load(s_model_path, map_location='cuda:{}'.format(parser_args.gpu))
        sparsity_rate = i/parser_args.sparse_epoch
        s_model = get_sparse_model(model, sparsity_rate)
        s_all_num, s_norm2, s_max_w = get_params_size(s_model)
        #R = (norm2 - s_norm2).cpu().detach()
        R = get_params_diff(model, s_model).cpu().detach()
        R_list.append(R)
        acc_list.append(csv_data['acc'][i])
        sparse_rate_list.append(i/parser_args.sparse_epoch)

    sparse_rate_pr_gt_diff_list = []
    for i in range(len(sparse_rate_list)):
        R = int(R_list[i].round())
        print(R)
        sparse_rate_gt = sparse_rate_list[i]
        sparse_rate_pr = gw_list[R]
        sparse_rate_pr_gt_diff = abs(sparse_rate_pr - sparse_rate_gt)
        sparse_rate_pr_gt_diff_list.append(sparse_rate_pr_gt_diff)
    
    pr_sparse_rate_index = sparse_rate_pr_gt_diff_list.index(min(sparse_rate_pr_gt_diff_list))
    sparse_rate = pr_sparse_rate_index/parser_args.sparse_epoch
    print(sparse_rate)
    sparse_index = pr_sparse_rate_index

    acc_loss_dict = {}
    acc_loss_dict["R_list"] = R_list
    acc_loss_dict["sparse_rate_list"] = sparse_rate_list
    acc_loss_dict["gw_list"] = gw_list
    acc_loss_dict["sparse_index"] = sparse_index
    acc_loss_dict["norm2"] = norm2
    np.save(parser_args.file_path + 'acc_loss_dict.npy', acc_loss_dict)

    plt.clf()
    plt.figure(dpi=400,figsize=(6,4))
    plt.plot(R_list, sparse_rate_list, label='Network Actual Value', c='b')
    plt.plot(range(0,int(norm2)+1), gw_list, label='Theory Prediction', c='g')
    plt.xlabel("R")
    plt.ylabel("Pruning Ratio)")
    plt.grid()
    plt.legend()
    plt.savefig(parser_args.file_path + 'predict.jpg', bbox_inches='tight')
    plt.close('all')

    train_csv_data = pd.read_csv(parser_args.file_path + 'sparse_train/acc_loss.csv')

    plt.clf()
    plt.figure(dpi=400,figsize=(6,4))
    if ((max(csv_data['loss'])/csv_data['loss'][0])>5) or ((max(train_csv_data['loss'])/train_csv_data['loss'][0])) > 5:
        plt.plot(sparse_rate_list, [math.log10(item/csv_data['loss'][0])+2 for item in csv_data['loss']], label='test_sparse_loss_log10', c='b')
        plt.plot(sparse_rate_list, [math.log10(item/train_csv_data['loss'][0])+1.5 for item in train_csv_data['loss']], label='train_sparse_loss_log10', c='g')
    else:
        plt.plot(sparse_rate_list, [item/csv_data['loss'][0] for item in csv_data['loss']], label='test_sparse_loss', c='b')
        plt.plot(sparse_rate_list, [item/train_csv_data['loss'][0] for item in train_csv_data['loss']], label='train_sparse_loss', c='g')
    plt.plot(sparse_rate_list, [item/100 for item in csv_data['acc']], label='test_sparse_acc', c='r')
    plt.plot(sparse_rate_list, [item/100 for item in train_csv_data['acc']], label='train_sparse_acc', c='k')
    plt.axvline(sparse_index/parser_args.sparse_epoch)
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Accuracy & Loss")
    plt.grid()
    plt.legend()
    plt.savefig(parser_args.file_path + 'acc_loss.jpg', bbox_inches='tight')
    plt.close('all')

    return sparse_index