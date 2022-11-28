import torch
import torch.nn as nn
import numpy as np


class ReparameterizedMVN(nn.Module):
    def __init__(
            self, 
            n_patch, 
            n_z
            ):
        super(ReparameterizedMVN, self).__init__()
        """
        L : std 
        """
        self.n_z = n_z
        self.n_patch = n_patch
        self.m = nn.Parameter(torch.zeros((1,n_patch, n_z)))
        self.L = nn.Parameter(
                torch.eye(n_z).repeat(n_patch, 1, 1)
                )

    def forward(self,z):
        """
        args : randomly rampled MVN random variable
        return : mean + std * rv
        """
        return self.m + \
                torch.einsum('bpz, pzk->bpk',
                z, 
                self.L.transpose(1,2)
                )

    def logp(self, x):
        """
        input : x = log of the probability distribution (MVN)
        """
        C = torch.einsum('pzk, pkl->pzl', self.L, self.L.transpose(1,2))
        return torch_mvn_logp(x, self.m, C)

    def entropy(self):
        C = torch.einsum('pzk, pkl->pzl', self.L, self.L.transpose(1,2))
        H = (1/2) * torch.logdet(2*np.pi * np.e * C).mean()
        return H

    def sample(self, N):
        return self(torch.randn(N, self.n_patch, self.n_z).to(self.m.device))


def torch_mvn_logp(x, m, C):
    """
    Input
        x : (bs, pat, k) data
        m : (1, pat, k) mean of the data
        C : (pat, k,k) cov of the data
    output
        (N,) log p = N(x; m, c) 
    """
    assert len(x.shape)==3
    k = x.shape[2]
    Z = -(k/2) * np.log(2*np.pi) - (1/2) * torch.logdet(C)
    # shape of Z : (pat,)
    Cinv = torch.inverse(C)
    ## shape of C : (pat, k, k)
    ## shape of x-m : (bs, pat, k)
    s = -(1/2) *(
            torch.einsum('bpk,pkl->bpl',(x-m), Cinv) * (x-m)
            ).sum(-1)

    return Z + s






