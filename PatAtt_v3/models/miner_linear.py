import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.flow import Glow


def load_flow(inp_dim, hidden_channels, K, sn, nonlin, flow_permutation):
    glow_default = {'mlp': True,
                    'image_shape': None,
                    'actnorm_scale': 1,
                    'flow_coupling': 'additive',
                    'LU_decomposed': True,
                    'y_classes': -1,
                    'L': 0, # Not used for MLP
                    'learn_top': False,
                    'y_condition': False,
                    'logittransform': False,
                    'use_binning_correction': False,
                    'use_actnorm': False
                    }
    flow = Glow(inp_dim=inp_dim,
                hidden_channels=hidden_channels,
                K=K,
                sn=sn,
                nonlin=nonlin,
                flow_permutation=flow_permutation,
                **glow_default)
    flow.return_ll_only = True
    return flow

def load_glow(inp_dim, hidden_channels, K, sn, nonlin, flow_permutation, flow_coupling, flow_L, use_actnorm):
    glow_default = {'mlp': False,
                    'actnorm_scale': 1,
                    'LU_decomposed': True,
                    'y_classes': -1,
                    'learn_top': False,
                    'y_condition': False,
                    'logittransform': False,
                    'use_binning_correction': False,
                    }
    flow = Glow(inp_dim=None,
                image_shape=(1, 1, inp_dim),
                hidden_channels=hidden_channels,
                K=K,
                sn=sn,
                nonlin=nonlin,
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                L=flow_L,
                use_actnorm=use_actnorm,
                **glow_default)
    flow.return_ll_only = True
    return flow


class FlowMiner(nn.Module):
    def __init__(self, nz0, flow_permutation, K, flow_glow=False, flow_coupling='additive', flow_L=1, flow_use_actnorm=True):
        super(FlowMiner, self).__init__()
        self.nz0 = nz0
        self.is_glow = flow_glow
        if flow_glow:
            self.flow = load_glow(inp_dim=self.nz0,
                                  hidden_channels=100,
                                  K=K,
                                  sn=False,
                                  nonlin='elu',
                                  flow_permutation=flow_permutation,
                                  flow_coupling=flow_coupling,
                                  flow_L=flow_L,
                                  use_actnorm=flow_use_actnorm
                                  )
            self.flow.cuda()
            # Init Actnorm
            init_z = torch.randn(100, self.nz0, 1, 1).cuda()
            self.flow(init_z)
        else:
            self.flow = load_flow(inp_dim=self.nz0,
                                  hidden_channels=100,
                                  K=K,
                                  sn=False,
                                  nonlin='elu',
                                  flow_permutation=flow_permutation
                                  )
           

    def forward(self, z):
        if self.is_glow:
            z = z.unsqueeze(-1).unsqueeze(-1)
        z0 = self.flow.reverse_flow(z, y_onehot=None, temperature=1)
        if self.is_glow:
            z0 = z0.squeeze(-1).squeeze(-1)
        return z0

    def logp(self, x):
        if self.is_glow:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.flow(x)

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.flow.set_actnorm_init()

class ReparameterizedGMM_Linear(nn.Module):
    def __init__(
            self,
            n_patch,
            n_z,
            n_components=10):
        super(ReparameterizedGMM_Linear, self).__init__()
        self.n_z = n_z
        self.n_patch = n_patch
        self.n_components = n_components
        self.mvns = [ReparameterizedMVN_Linear(n_patch = n_patch, n_z = n_z) for _ in range(self.n_components)]
        for ll, mvn in enumerate(self.mvns):
            mvn.m.data = torch.randn_like(mvn.m.data)
            for name, p in mvn.named_parameters():
                self.register_parameter(f'mvn_{ll}_{name}', p)
        self.mixing_weight_logits = nn.Parameter(torch.zeros(self.n_components))

    def sample_components_onehot(self, n):
        #return F.gumbel_softmax(self.mixing_weight_logits[None].repeat(n,1), hard=True)
        return F.one_hot(torch.multinomial(F.softmax(self.mixing_weight_logits[None]).view(-1), n, replacement=True), num_classes = self.n_components)

    def forward(self, z):
        bs = z.shape[0]
        masks = self.sample_components_onehot(bs)
        masks = masks.t()
        samps = torch.stack([mvn(z).reshape([bs,-1]) for mvn in self.mvns])

        x = (masks[..., None] * samps).sum(0)
        return x.view([bs, self.n_patch, self.n_z]) 
    
    def logp(self,x):
        n = x.shape[0]
        logps = []
        for mvn in self.mvns:
            logp = mvn.logp(x.view(n, -1))
            logps.append(logp)
        logps = torch.stack(logps)
        log_mixing_weights = F.log_softmax(self.mixing_weight_logits[None].repeat(n, 1), dim=1).t()
        logp = torch.logsumexp(logps + log_mixing_weights, dim=0)
        return logp

    def sample(self, N):
        return self(torch.randn(N, self.n_patch * self.n_z).to(self.mvns[0].m.device))

    
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
        self.m = nn.Parameter(torch.randn((1,n_patch * n_z)))
        self.L = nn.Parameter(
                torch.eye(n_z * n_patch)
                )
#        self.mask = torch.tril(
#                torch.ones_like(self.L))

    def forward(self,z):
        """
        args : randomly rampled MVN random variable
        return : mean + std * rv
        """
#        L = torch.tril(self.L)
        return (self.m + z.view([-1, self.n_patch * self.n_z]) @ self.L.T).reshape([-1, self.n_patch, self.n_z])

    def logp(self, x):
        """
        input : x = log of the probability distribution (MVN)
        """
#        L = torch.tril(self.L)
        C = self.L @ self.L.T 
        return torch_mvn_logp_linear(x, self.m, C)

    def entropy(self):
#        L = torch.tril(self.L)
        C = self.L @ self.L.T 
        H = (1/2) * torch.logdet(2*np.pi * np.e * C+1e-8)
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
#    assert len(x.shape)==3
    k = x.shape[1]
    Z = -(k/2) * np.log(2*np.pi) - (1/2) * torch.logdet(C+1e-8)
    # shape of Z : (pat,)
    Cinv = torch.inverse(C+1e-8)
    ## shape of C : (k, k)
    ## shape of x-m : (bs, pat, k)
    s = -(1/2) *(
            ((x-m) @ Cinv) * (x-m)).sum(-1)

    return Z + s




def gaussian_logp(mean, logstd, x, detach=False):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
    k = 1 (Independent)
    Var = logstd ** 2
    """
    c = np.log(2 * np.pi)
    v = -0.5 * (logstd * 2. + ((x - mean) ** 2) / torch.exp(logstd * 2.) + c)
    if detach:
        v = v.detach()
    return v


