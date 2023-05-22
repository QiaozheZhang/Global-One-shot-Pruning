import torch, time
import math
from torch.autograd import Variable
import numpy as np

from hessian_utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
from utils import *
from args import *


class hessian():
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda:{}'.format(parser_args.gpu)
            self.model = model.eval()  # make model is in evaluation model
        else:
            self.device = 'cpu'
            self.model = model.to(self.device).eval()  # make model is in evaluation model

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if 'cuda' in self.device:
                self.inputs, self.targets = self.inputs.cuda(parser_args.gpu
                ), self.targets.cuda(parser_args.gpu)

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            if parser_args.l1:
                loss = self.criterion(outputs, self.targets.to(self.device)) + get_regularization_loss_my(self.model, regularizer=parser_args.regularizer,lmbda=parser_args.lmbda)
            else:
                loss = self.criterion(outputs, self.targets.to(self.device))
            loss.backward(create_graph=True)

        for n, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.grad = None
                m.weight.grad = None
                m.bias.grad = None
                #(m.requires_grad, m.weight.requires_grad, m.bias.requires_grad)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            if parser_args.model == 'AlexNet':
                inputs = F.interpolate(inputs, scale_factor=7)
            outputs = self.model(inputs.to(device))
            if parser_args.l1:
                loss = self.criterion(outputs, targets.to(device)) + get_regularization_loss_my(self.model, regularizer=parser_args.regularizer,lmbda=parser_args.lmbda)
            else:
                loss = self.criterion(outputs, targets.to(device))
            #print('calculate loss')
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
                    print(eigenvalue, computed_dim, i)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1, min_iter=True):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                print(i, iter)
                if i == 0:
                    one_time = time.time()
                    two_time = time.time()
                else:
                    one_time = two_time
                    two_time = time.time()
                    diff_time = two_time - one_time
                    need_time = (iter-i)*diff_time/(3600)
                    print("need time {} hours".format(need_time))
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

                if min_iter:
                    T1 = torch.zeros(i+1, i+1).to(device)
                    for i in range(len(alpha_list)):
                        T1[i, i] = alpha_list[i]
                        if i < len(alpha_list) - 1:
                            T1[i + 1, i] = beta_list[i]
                            T1[i, i + 1] = beta_list[i]
                    a1_, b1_ = torch.eig(T1, eigenvectors=True)

                    sorted_list1 = sorted(a1_[:, 0], key=abs, reverse=True)
                    if i == 0:
                        last_min_ev =torch.tensor(0.)
                    min_ev = sorted_list1[-1]
                    diff_ev = abs(abs(min_ev)-abs(last_min_ev))
                    last_min_ev = min_ev
                    print(last_min_ev)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :]**2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
