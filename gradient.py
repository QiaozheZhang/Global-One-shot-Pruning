import torch, matplotlib, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from trainer import *
from hessian_utils import *

init_file_path()
train_loader, val_loader, actual_val_loader = get_dataset()
criterion = get_criterion()
parser_args.mode = 'sparse'
model_path = parser_args.file_path + '{}/model/model_{}.pth'.format(parser_args.mode, 0)
model = torch.load(model_path, map_location='cuda:{}'.format(parser_args.gpu))

def calcu(index):
    sparsity_rate = index / parser_args.sparse_epoch
    weight_threshold = get_sparse_threshold(model, sparsity_rate)

    params, grads = get_params_grad(model)
    model.zero_grad()
    params_index = 0
    for j in range(len(params)):
        tmp_weight_threshold_index = torch.nonzero(torch.abs(params[j])==weight_threshold)
        if len(tmp_weight_threshold_index) == 1:
            params_index = j
            weight_threshold_index = tmp_weight_threshold_index[0]
            print(weight_threshold_index)
        elif len(tmp_weight_threshold_index) > 1:
            print("多个值等于threshold")
            params_index = j
            weight_threshold_index = tmp_weight_threshold_index[0]
            print(weight_threshold_index)

    threshold_grads_sum, threshold_H_sum, add_threshold_H = 0, 0, 0
    num_data = 0
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
        ):

        images = images.cuda(parser_args.gpu, non_blocking=True)
        target = target.cuda(parser_args.gpu, non_blocking=True)
        if parser_args.model == 'AlexNet':
                images = F.interpolate(images, scale_factor=7)
        output = model(images)
        if parser_args.criterion in ['mse']:
            label = torch.nn.functional.one_hot(target, num_classes=np.shape(output)[1]).to(torch.float)
        else:
            label = target
        loss = criterion(output, label)

        if parser_args.l1:
            regularization_loss = torch.tensor(0)
            regularization_loss = get_regularization_loss_my(model, regularizer=parser_args.regularizer, lmbda=parser_args.lmbda)
            loss += regularization_loss

        loss.backward(create_graph=True)
        params, grads = get_params_grad(model)
        model.zero_grad()

        threshold_grads = grads[params_index]
        for m in range(len(weight_threshold_index)):
            threshold_grads = threshold_grads[weight_threshold_index[m]]

        threshold_H_list = torch.autograd.grad(threshold_grads,
                                             params,
                                             only_inputs=True,
                                             retain_graph=False)

        threshold_H = threshold_H_list[params_index]
        for n in range(len(weight_threshold_index)):
            threshold_H = threshold_H[weight_threshold_index[n]]
        
        for z in range(len(threshold_H_list)):
            add_threshold_H += torch.sum(torch.abs(threshold_H_list[z]))

        tmp_num_data = images.size(0)
        threshold_grads_sum += threshold_grads*tmp_num_data
        threshold_H_sum += threshold_H*tmp_num_data
        num_data += float(tmp_num_data)
    print(threshold_grads_sum/num_data, threshold_H/num_data, add_threshold_H/num_data)
    model.zero_grad()

    return threshold_grads_sum/num_data, threshold_H/num_data, add_threshold_H/num_data

def gradie():
    threshold_grads_list, threshold_H_list, add_threshold_list = [], [], []
    log_threshold_grads_list, log_threshold_H_list, log_add_threshold_list = [], [], []
    for i in range(100):
        print(i)
        index = i * 10
        threshold_grads, threshold_H, add_threshold = calcu(index)
        threshold_grads_list.append(threshold_grads.detach().cpu())
        threshold_H_list.append(threshold_H.detach().cpu())
        add_threshold_list.append(add_threshold.detach().cpu())

        log_threshold_grads_list.append(math.log10(abs(threshold_grads.detach().cpu())+1e-40))
        log_threshold_H_list.append(math.log10(abs(threshold_H.detach().cpu()+1e-40)))
        log_add_threshold_list.append(math.log10(abs(add_threshold.detach().cpu()+1e-40)))
    
    gradient_dict = {}
    gradient_dict["threshold_grads_list"] = threshold_grads_list
    gradient_dict["threshold_H_list"] = threshold_H_list
    gradient_dict["add_threshold_list"] = add_threshold_list
    np.save(parser_args.file_path + 'gradient_dict.npy', gradient_dict)

gradie()

gradient_dict = np.load(parser_args.file_path + 'gradient_dict.npy', allow_pickle=True).item()
add_threshold_list = gradient_dict["add_threshold_list"]

num = 0
for item in add_threshold_list:
    if item < 1e-7:
        num+=1
    print(item)
print(num)

plt.clf()
plt.figure(dpi=400,figsize=(7.5,5))
plt.scatter(range(len(add_threshold_list)), [item for item in add_threshold_list], label='Row L1 Norm', c='b', marker="*")
plt.subplots_adjust(bottom=0.15,left=0.15)
fontsize = 16
plt.xticks(size=fontsize)
plt.yticks(size=fontsize)
plt.xlabel("Hessian Matrix Index(%)", fontsize=fontsize)
plt.ylabel("Row L1 Norm in Hessian Matrix", fontsize=fontsize)
plt.grid()
plt.legend(fontsize=fontsize)
plt.savefig(parser_args.file_path + 'hessian.eps', format='eps')
plt.savefig(parser_args.file_path + 'hessian.jpg')
plt.close('all')