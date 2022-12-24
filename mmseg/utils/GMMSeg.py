import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def shifted_var(tensor, rowvar=True, bias=True):
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    # * input have already been shifted
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    # tensor: d,n
    var = (tensor ** 2).sum(-1)
    return factor * var

# * uniform marginal distribution
@torch.no_grad()
def distributed_sinkhorn_wograd(out, sinkhorn_iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t() # K x B

    B = Q.shape[1] # * num_pixels
    K = Q.shape[0] # * num_components

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q = Q/sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per component must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q/sum_of_rows
        Q = Q/K # * it should be *true_distribution, and in this form, it is 

        # normalize each column: total weight per sample must be 1/B
        Q = Q/torch.sum(Q, dim=0, keepdim=True)
        Q = Q/B

    Q = Q*B # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()

    return Q, indexs

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update

@torch.no_grad()
def rnd_sample(pop_size, num_samples, _uniform=False, _device=None):
    if _uniform:
        return torch.linspace(0, pop_size-1, num_samples, dtype=torch.int64, device=_device)
    else:
        return torch.randperm(pop_size, dtype=torch.int64)[:num_samples]