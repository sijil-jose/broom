import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)





def stable_inverse(cov, initial_jitter=1e-6, max_tries=5):
    n = cov.size(-1)
    I = torch.eye(n, device=cov.device)
    # enforce symmetry
    #cov = 0.5 * (cov + cov.transpose(-1, -2))
    jitter = initial_jitter
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(cov + jitter * I)
            return torch.cholesky_inverse(L)
        except RuntimeError:
            print('fkjfkmfl')
            jitter *= 10


def compute_loss(logits, map_in_nl, noi_cov, gnilc_in_nl, ncluster, temperature=0.5):
    # 1. Gumbel-Softmax (soft for differentiability)
    #part = F.gumbel_softmax(logits, tau=temperature, hard=True)  # [npix, ncluster]
    nfreq = 22
    part_soft = F.softmax(logits / temperature, dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    a = torch.ones(nfreq, dtype = torch.float64, device =device)
    softmax = nn.Softmax(dim=1)

    # hard assignment
    hard_idx = part_soft.argmax(dim=1)
    part_hard = F.one_hot(hard_idx, ncluster).to(torch.float64)

    # straight-through estimator
    part = part_hard + (part_soft - part_soft.detach())
    
    
    ## covariance from input maps
    # cluster sizes
    Nk = part.sum(dim=0)  # [ncluster]
    # cluster mean
    mu_k = torch.einsum("nk,nf->kf", part, map_in_nl) / Nk[:, None]
    # cluster second moment
    Q_k = torch.einsum("nk,nf,ng->kfg", part, map_in_nl, map_in_nl) / Nk[:, None, None]

    # covariance per cluster
    cov_k = Q_k - mu_k.unsqueeze(2) @ mu_k.unsqueeze(1)  # [ncluster, nfreq, nfreq]
    '''
    ## Covariance from gnilc maps
    # cluster sizes
    Nk = part.sum(dim=0)  # [ncluster]
    # cluster mean
    mu_k = torch.einsum("nk,nf->kf", part, gnilc_in_nl) / Nk[:, None]
    # cluster second moment
    Q_k = torch.einsum("nk,nf,ng->kfg", part, gnilc_in_nl, gnilc_in_nl) / Nk[:, None, None]

    # covariance per cluster
    cov_k = Q_k - mu_k.unsqueeze(2) @ mu_k.unsqueeze(1)  # [ncluster, nfreq, nfreq]
    '''
    # regularize
    eye = torch.eye(nfreq, device=device, dtype=torch.float64)
    cov_k = cov_k + 1e-6 * eye

    inv_cov_k = torch.linalg.inv(cov_k)

    num = torch.einsum("kfg,g->kf", inv_cov_k, a)
    den = torch.einsum("kf,f->k", num, a)

    w_k = num / den[:, None]

    # assign each pixel its cluster weight
    w_pixel = torch.einsum("nk,kf->nf", part, w_k)  # [npix, nfreq]

    noi_var = torch.einsum('bi,ij,bj->b', w_k, noi_cov, w_k).mean()
    gnilc_out = (w_pixel * gnilc_in_nl).sum(dim=1)
    #gnilc_out = (w_pixel * map_in_nl).sum(dim=1)

    return gnilc_out.var(), noi_var, w_pixel

