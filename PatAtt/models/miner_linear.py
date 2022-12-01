import torch
import torch.nn as nn
import numpy as np


class ReparameterizedMVN_Linear(nn.Module):
    def __init__(
            self, 
            n_patch, 
            n_z
            ):
        super(ReparameterizedMVN_Linear, self).__init__()
        """
        L : std 
        """
        self.n_z = n_z
        self.n_patch = n_patch
        self.m = nn.Parameter(torch.zeros((1,n_patch * n_z)))
        self.L = nn.Parameter(
                torch.eye(n_z * n_patch)
                )

    def forward(self,z):
        """
        args : randomly rampled MVN random variable
        return : mean + std * rv
        """
        return (self.m + z.view([-1, self.n_patch * self.n_z]) @ self.L.T).reshape([-1, self.n_patch, self.n_z])

    def logp(self, x):
        """
        input : x = log of the probability distribution (MVN)
        """
        C = self.L @ self.L.T 
        return torch_mvn_logp_linear(x, self.m, C)

    def entropy(self):
        C = self.L @ self.L.T 
        H = (1/2) * torch.logdet(2*np.pi * np.e * C)
        return H

    def sample(self, N):
        return self(torch.randn(N, self.n_patch * self.n_z).to(self.m.device))


def torch_mvn_logp_linear(x, m, C):
    """
    Input
        x : (bs, pat * n_z) data
        m : (1, pat * n_z) mean of the data
        C : (pat* n_z, pat * n_z) cov of the data
    output
        (N,) log p = N(x; m, c) 
    """
    assert len(x.shape)==3
    k = x.shape[1]
    Z = -(k/2) * np.log(2*np.pi) - (1/2) * torch.logdet(C)
    # shape of Z : (pat,)
    Cinv = torch.inverse(C)
    ## shape of C : (k, k)
    ## shape of x-m : (bs, pat, k)
    s = -(1/2) *(
            ((x-m) @ Cinv) * (x-m)).sum(-1)

    return Z + s






