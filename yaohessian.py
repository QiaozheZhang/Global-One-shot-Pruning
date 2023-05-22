import numpy as np
import torch 
from torchvision import datasets, transforms
from my_pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot

import matplotlib.pyplot as plt

import matplotlib, math
import tqdm, sys
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from utils import *
from trainer import *
import os

def get_density(iter_num=128, n_v_num=2, index=0):
    init_file_path()
    train_loader, val_loader, actual_val_loader = get_dataset()
    criterion = get_criterion()
    parser_args.mode = 'sparse'

    model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, index)
    model = torch.load(model_path, map_location='cuda:{}'.format(parser_args.gpu))

    hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=True)

    eigen_list_path = parser_args.file_path + '{}/hessien_eigenvalue/{}_{}_eigen.pth'.format(parser_args.mode, iter_num, index)
    weight_list_path = parser_args.file_path + '{}/hessien_eigenvalue/{}_{}_weight.pth'.format(parser_args.mode, iter_num, index)
    check_dir(parser_args.file_path + '{}/hessien_eigenvalue/'.format(parser_args.mode))
    
    density_eigen, density_weight = hessian_comp.density(iter=iter_num,n_v=n_v_num)
    torch.save(density_eigen, eigen_list_path)
    torch.save(density_weight, weight_list_path)
    density_eigen = torch.load(eigen_list_path)
    density_weight = torch.load(weight_list_path)
    sorted_eigen_list = sorted(density_eigen[0], key=abs)
    eigen_path = parser_args.file_path + '{}/hessien_eigenvalue/{}_{}.txt'.format(parser_args.mode, iter_num, index)
    f = open(eigen_path,"w")
    for line in sorted_eigen_list:
        f.write(str(line)+'\n')
    f.close()

    get_esd_plot(density_eigen, density_weight)

def get_trace(index=0):
    init_file_path()
    train_loader, val_loader, actual_val_loader = get_dataset()
    criterion = get_criterion()
    parser_args.mode = 'sparse'

    model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, index)
    model = torch.load(model_path, map_location='cuda:{}'.format(parser_args.gpu))

    hessian_comp = hessian(model, criterion, dataloader=train_loader, cuda=True)

    trace = 0
    trace = hessian_comp.trace()
    print("The trace of this model is: %.4f"%(np.mean(trace)))

    return np.mean(trace)