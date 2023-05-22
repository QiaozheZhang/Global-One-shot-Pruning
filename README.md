# How Sparse Can We Prune a Deep Network

## Project Info
This project is the official implementation of the paper "How Sparse Can We Prune a Deep Network: A Geometric Viewpoint".

## Overview

---

Overparameterization constitutes one of the most significant  hallmarks of deep neural networks. Though it can offer the advantage of outstanding generalization performance, it meanwhile imposes substantial storage burden, thus necessitating the study of network pruning.  A natural and  fundamental question is: How sparse can we prune a deep network (with almost no hurt on the performance)?  To address this problem, in this work we take a first principles approach,  specifically, by merely enforcing the sparsity constraint on the original loss function, we're able to characterize the sharp phase transition point of pruning ratio, which corresponds to the  boundary between the feasible and the infeasible, from the perspective of high-dimensional geometry.  It turns out that the phase transition point of pruning ratio equals the squared Gaussian width of some convex body resulting from the \(l_1\)-regularized loss function,  normalized by the original dimension of parameters. As a byproduct, we provide a novel network pruning algorithm which is essentially a global one-shot pruning one. Furthermore, we provide efficient countermeasures to address the challenges in computing the involved Gaussian width, including the spectrum estimation of a large-scale Hessian matrix and dealing with the non-definite positiveness of a Hessian matrix.  It is demonstrated that the predicted pruning ratio threshold coincides very well with the actual value obtained from the experiments and our proposed pruning algorithm can achieve competitive or even better performance than the existing pruning algorithms.

## Installation

---

### 1. Requirements

* Python 3.8.10
* Pytorch 1.12.1
* torchvision 0.13.1
* CUDA 11.6.0
* cuDNN 8
* tqdm

---

Have tested on nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04 with RTX3090, Intel 10920X and 196G Ram.

**Note: the model is saved in each step of the training and sparse process, this project takes up a lot of disk space, please always pay attention to the remaining capacity of the disk to avoid quitting the experiment** 

### 2. Train Network & Get Sparse Model

```bash
python main.py --config path_to_the_config
```

### 3. Calculate the "important eigenvalues"

```bash
python gradient.py --config path_to_the_config
```

**Note:** important eigenvalues calculation is only suitable for large networks including **Alexnet, VGG16, ResNet18 and ResNet50**.

### 4.Calculate the Gaussain Width

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

