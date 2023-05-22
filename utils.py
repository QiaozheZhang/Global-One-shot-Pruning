import os, copy, matplotlib
import tqdm, sys
import torch, torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim.lr_scheduler import _LRScheduler

from args import *
from model import *
from hessian_utils import *

def check_dir(dir):
    if os.path.exists(dir) != True:
        os.makedirs(dir)

def get_dataset():
    if parser_args.dataset in ['cifar10']:
        data_root = os.path.join(parser_args.data_path, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        if parser_args.use_full_data:
            train_dataset = dataset
            validation_dataset = test_dataset
        else:
            val_size = 5000
            train_size = len(dataset) - val_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
        )

        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=parser_args.batch_size, shuffle=False, **kwargs
        )

        actual_val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=parser_args.batch_size, shuffle=True, **kwargs
        )
    
    elif parser_args.dataset in ['mnist']:
        data_root = os.path.join(parser_args.data_path, "mnist")
        use_cuda = torch.cuda.is_available()
        
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=parser_args.batch_size,
            shuffle=True,
            **kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=parser_args.batch_size,
            shuffle=True,
            **kwargs
        )
        actual_val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=parser_args.batch_size,
            shuffle=True,
            **kwargs
        )
    
    elif parser_args.dataset in ['cifar100']:
        data_root = os.path.join(parser_args.data_path, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, parser_args.batch_size, shuffle=True, **kwargs)

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        val_loader = torch.utils.data.DataLoader(test_dataset, parser_args.batch_size, shuffle=True, **kwargs)
        actual_val_loader = torch.utils.data.DataLoader(test_dataset, parser_args.batch_size, shuffle=True, **kwargs)
    
    return train_loader, val_loader, actual_val_loader

def get_model():
    if parser_args.model in ['FC']:
        model = FC()
    if parser_args.model in ['FC2']:
        model = FC2()
    elif parser_args.model in ['FC12']:
        model = FC12()
    elif parser_args.model in ['FC13']:
        model = FC13()
    elif parser_args.model in ['LeNet']:
        model = LeNet()
    elif parser_args.model in ['CNet']:
        model = CNet()
    elif parser_args.model in ['AlexNet']:
        model = AlexNet()
    elif 'VGG' in parser_args.model:
        model = VGG(parser_args.model)
    elif parser_args.model in ['ResNet18']:
        model = resnet18()
    elif parser_args.model in ['ResNet50']:
        model = resnet50()
    else:
        print('no model !!!!')

    model.initialize_weights()

    return model

def get_criterion():
    if parser_args.criterion in ['cross_entropy']:
        criterion = nn.CrossEntropyLoss().cuda(parser_args.gpu)
    if parser_args.criterion in ['mse']:
        criterion = nn.MSELoss().cuda(parser_args.gpu)

    return criterion

def get_optimizer(model):
    parameters = list(model.named_parameters())
    bn_params = [v for n, v in parameters if (
        "bn" in n) and v.requires_grad]
    rest_params = [v for n, v in parameters if (
        "bn" not in n) and v.requires_grad]
    optimizer = torch.optim.SGD(
        [
            {
                "params": bn_params,
                "weight_decay": 0 if False else parser_args.weight_decay,
            },
            {"params": rest_params, "weight_decay": parser_args.weight_decay},
        ],
        parser_args.lr,
        momentum=parser_args.momentum,
        weight_decay=parser_args.weight_decay,
        nesterov=False,
    )

    return optimizer

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def get_scheduler(optimizer, milestones=[80, 120], gamma=0.5, max_epochs=150):
    gamma = parser_args.gamma
    milestones = parser_args.milestones
    max_epochs = parser_args.epoch

    """ if parser_args.epoch in [6]:
        milestones = [3,]
        gamma = parser_args.lr_gamma
    elif parser_args.epoch == 100:
        milestones = [50, 80]
        max_epochs = 100
    elif parser_args.epoch in [150, 160]:
        milestones = [80, 120]
        max_epochs = parser_args.epoch
    elif parser_args.epoch == 200:
        milestones = [100, 150]
        max_epochs = 200
    elif parser_args.epoch == 300:
        milestones = [150, 250]
        max_epochs = 300
    else:
        max_epochs = parser_args.epoch
        milestones = [300, 350, 400, 450, 470] """
    
    print("##############################", milestones, parser_args.schedumode)
    if parser_args.schedumode == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif parser_args.schedumode == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    return scheduler

def get_layers(model):
    if parser_args.model in ['FC', 'FC2', 'FC12', 'FC13', 'LeNet', 'VGG16', 'CNet', 'AlexNet']:
        linear_layers, conv_layers = [], []
        for layer_index in range(0, len(model.linear)):
            if isinstance(model.linear[layer_index], nn.Linear):
                linear_layers.append(model.linear[layer_index])

        for layer_index in range(0, len(model.conv)):
            if isinstance(model.conv[layer_index], nn.Conv2d):
                conv_layers.append(model.conv[layer_index])
    elif parser_args.model in ['ResNet18', 'ResNet50']:
        linear_layers = [model.fc]
        conv_layers = []
        for layer in model.conv1:
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)
        for layer in [model.conv2_x, model.conv3_x, model.conv4_x, model.conv5_x]:
            for basic_block_id in range(len(layer)):
                for basic_layer in layer[basic_block_id].residual_function:
                    if isinstance(basic_layer, nn.Conv2d):
                        conv_layers.append(basic_layer)
                for basic_layer in layer[basic_block_id].shortcut:
                    if isinstance(basic_layer, nn.Conv2d):
                        conv_layers.append(basic_layer)

    return linear_layers, conv_layers

def get_regularization_loss_my(model, regularizer='L1', lmbda=1):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    linear_layers, conv_layers = get_layers(model)

    regularization_loss = torch.tensor(0.).cuda(parser_args.gpu)
    if regularizer == 'L2':
        for name, params in model.named_parameters():
            if ".bias" in name:
                if ".bias_flag" in name:
                    pass
                elif ".bias_score" in name:
                    pass
                else:
                    regularization_loss += torch.norm(params, p=2)**2

            elif ".weight" in name:
                regularization_loss += torch.norm(params, p=2)**2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'L1':
        # reg_loss =  ||p||_1
        for name, params in model.named_parameters():
            if ".bias" in name:
                if ".bias_flag" in name:
                    pass
                elif ".bias_score" in name:
                    pass
                else:
                    regularization_loss += torch.norm(params, p=1)

            elif ".weight" in name:
                regularization_loss += torch.norm(params, p=1)
        regularization_loss = lmbda * regularization_loss

    #print('red loss: ', regularization_loss)

    return regularization_loss

def get_sparse_model(model, sparsity_rate):
    cp_model = copy.deepcopy(model)
    linear_layers, conv_layers = get_layers(cp_model)

    num_active_weight = 0
    num_active_biases = 0
    active_weight_list = []
    active_bias_list = []

    for layer in conv_layers+linear_layers:
        num_active_weight += torch.ones_like(layer.weight).data.sum().item()
        active_weight = layer.weight.data.clone()
        active_weight_list.append(active_weight.view(-1))

    number_of_weight_to_prune = np.ceil(
        sparsity_rate * num_active_weight).astype(int)
    number_of_biases_to_prune = np.ceil(
        sparsity_rate * num_active_biases).astype(int)

    agg_weight = torch.cat(active_weight_list)
    agg_bias = torch.tensor([])

    if number_of_weight_to_prune == 0:
        number_of_weight_to_prune = 1
    if number_of_biases_to_prune == 0:
        number_of_biases_to_prune = 1

    weight_threshold = torch.sort(
        torch.abs(agg_weight), descending=False).values[number_of_weight_to_prune-1].item()
    bias_threshold = -1

    num = 0
    for layer in conv_layers+linear_layers:
        scores = torch.gt(layer.weight.abs(), 
                           torch.ones_like(layer.weight)*weight_threshold).int()
    
        layer.weight.data = layer.weight.data * scores.data
        num += scores.data.sum().item()
    
    print('################################################################', weight_threshold, num_active_weight*sparsity_rate, num_active_weight-num)

    return cp_model

def get_sparse_threshold(model, sparsity_rate):
    cp_model = copy.deepcopy(model)
    linear_layers, conv_layers = get_layers(cp_model)

    num_active_weight = 0
    num_active_biases = 0
    active_weight_list = []
    active_bias_list = []

    for layer in conv_layers+linear_layers:
        num_active_weight += torch.ones_like(layer.weight).data.sum().item()
        active_weight = layer.weight.data.clone()
        active_weight_list.append(active_weight.view(-1))

    number_of_weight_to_prune = np.ceil(
        sparsity_rate * num_active_weight).astype(int)
    number_of_biases_to_prune = np.ceil(
        sparsity_rate * num_active_biases).astype(int)

    agg_weight = torch.cat(active_weight_list)
    agg_bias = torch.tensor([])

    if number_of_weight_to_prune == 0:
        number_of_weight_to_prune = 1
    if number_of_biases_to_prune == 0:
        number_of_biases_to_prune = 1

    weight_threshold = torch.sort(
        torch.abs(agg_weight), descending=False).values[number_of_weight_to_prune-1].item()
    
    print('################################################################', weight_threshold)

    return weight_threshold

def write_to_csv(acc1_list):
    file_path = parser_args.file_path + '{}/'.format(parser_args.mode)
    check_dir(file_path)

    results_df = pd.DataFrame({'acc':acc1_list})

    results_df_filename = file_path + 'result.csv'
    results_df.to_csv(results_df_filename, index=False)

def init_file_path():
    parser_args.file_path = './result/{}/{}/optim_{}_loss_{}_batch_size_{}_init_{}_lr_{}_momentum_{}_wd_{}_l1_{}_lmbda_{}_schedu_{}_schedumode_{}/'.format(
        parser_args.dataset, parser_args.model, parser_args.optim, parser_args.criterion, parser_args.batch_size, parser_args.init, parser_args.lr, parser_args.momentum, 
        parser_args.weight_decay, parser_args.l1, parser_args.lmbda, parser_args.schedu, parser_args.schedumode)

def save_model(model, epoch):
    file_path = parser_args.file_path + '{}/model/'.format(parser_args.mode)
    check_dir(file_path)
    torch.save(model, file_path + 'model_{}.pth'.format(epoch))

def get_params_size(model):
    size_list = []
    max_list = []
    norm2 = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        size_list.append(param.size())
        norm2 += torch.norm(param, p=2)
        max_list.append(torch.max(param))

    all_num = 0
    for i in range(len(size_list)):
        temp_size = size_list[i]
        temp_all_size = 1
        for j in range(len(temp_size)):
            temp_all_size *= temp_size[j]
        all_num += temp_all_size
    
    return all_num, norm2, max(max_list)

def cal_loss(model, train_loader, criterion):
    total, all_loss = 0, 0
    loss_list = []
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

        regularization_loss = torch.tensor(0)
        if parser_args.l1:
            regularization_loss = torch.tensor(0)
            regularization_loss =get_regularization_loss_my(model, regularizer=parser_args.regularizer,lmbda=parser_args.lmbda)
            #loss += regularization_loss

        total += target.size(0)

        all_loss += loss.detach().cpu()*target.size(0)
        if parser_args.l1:
            loss_list.append(loss.detach().cpu()+regularization_loss.detach().cpu())
        else:
            loss_list.append(loss.detach().cpu())
    
    if parser_args.l1:
        loss_result = all_loss/total+regularization_loss
    else:
        loss_result = all_loss/total

    print(loss_result, np.std(loss_list), regularization_loss/parser_args.lmbda)
    return loss_result, np.std(loss_list)

def write_AL_to_csv(acc1_list, loss_list):
    file_path = parser_args.file_path + '{}/'.format(parser_args.mode)
    check_dir(file_path)

    results_df = pd.DataFrame({'acc':acc1_list, 'loss': loss_list})

    results_df_filename = file_path + 'acc_loss.csv'
    print(results_df_filename)
    results_df.to_csv(results_df_filename, index=False)