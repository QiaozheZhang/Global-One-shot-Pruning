# How Sparse Can We Prune A Deep Network: A Fundamental Limit Viewpoint

## Project Info
This project is the official implementation of the paper [How Sparse Can We Prune A Deep Network: A Fundamental Limit Viewpoint].
<!-- :(https://arxiv.org/abs/2306.05857) -->

## Prerequisites

* Python 3.8.10
* Pytorch 1.12.1
* torchvision 0.13.1
* CUDA 11.3.1
* cuDNN 8
* tqdm
* pandas
* pyyaml
* matplotlib
* numpy>=1.20

Have tested on docker `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04` with RTX3090, Intel 10920X, 196G RAM and a 4T ssd.

**Note: the model is saved in each step of the training and sparse process, this project takes up a lot of disk space, please always pay attention to the remaining capacity of the disk to avoid quitting the experiment** 

## Installation

### 1.Clone this repo:
```bash
git clone https://github.com/QiaozheZhang/Global-One-shot-Pruning.git
cd Global-One-shot-Pruning
```
### 2.Install dependencies

## Usage

### 2. Train Network & Get Sparse Model

```bash
python main.py --config path_to_the_config
```

### 3. Calculate the "important eigenvalues"

```bash
python gradient.py --config path_to_the_config
```

**Note:** important eigenvalues calculation is only suitable for large networks including **Alexnet, VGG16, ResNet18 and ResNet50**.

### 4. Calculate the Gaussain Width

```bash
python gaussian_width.py --config path_to_the_config
```

## Configs

Note that workflows are managed by using `.yml` files specified in the `configs/` directory. Please refer to them to create new configurations, e.g. `AlexNet_CIFAR10_l1.yml`.

## Sample Config

```
model: AlexNet
init: kaiming_normal
data_path: "datasets/"
dataset: cifar10

use_full_data: True

batch_size: 128
gpu: 0

criterion: cross_entropy

optim: sgd
lr: 0.01
momentum: 0.9
weight_decay: 0 
gamma: 0.5

warm: 5
schedu: True
schedumode: multistep
milestones: [60, 120, 160]

epoch: 200
l1: True
lmbda: 0.00003
regularizer: L1
mode: train
sparse: True
sparse_epoch: 1000

file_path: ' '

save_model: True

lanczos_iter: 96
lanczos_num: 1
```

