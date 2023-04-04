import torch.nn as nn
import torch

class Glow(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, use_actnorm, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, y_classes,
                 learn_top, y_condition,logittransform,sn,
                 use_binning_correction=True, mlp=False,nonlin='relu'):
        super().__init__()
        self.mlp = mlp
        self.flow = FlowNet(image_shape=image_shape,
                            hidden_channels=hidden_channels,
                            K=K,
                            L=L,
                            use_actnorm=use_actnorm,
                            actnorm_scale=actnorm_scale,
                            flow_permutation=flow_permutation,
                            flow_coupling=flow_coupling,
                            LU_decomposed=LU_decomposed,
                            logittransform=logittransform,
                            sn=sn,
                            nonlin=nonlin)

        self.register_buffer("prior_h",
                             torch.zeros([1,
                                self.flow.output_shapes[-1][1] * 2,
                                self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]))

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.y_classes = y_classes
        self.y_condition = y_condition
        self.use_binning_correction = use_binning_correction
        self.learn_top = learn_top
        self.return_ll_only = False

    def prior(self, data, y_onehot=None, batch_size=0):
        if self.mlp:
            if data is not None:
                h = self.prior_h.repeat(data.shape[0], 1)
            else:
                h = self.prior_h.repeat(batch_size, 1)
        else:
            if data is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
            else:
                # Hardcoded a batch size of 32 here
                h = self.prior_h.repeat(batch_size, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(data.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None,
                reverse=False, use_last_split=False, batch_size=0):
        if reverse:
            # ipdb.set_trace()
            assert z is not None or batch_size > 0
            return self.reverse_flow(z, y_onehot, temperature, use_last_split, batch_size)
        else:
            z, objective, y_logits = self.normal_flow(x, y_onehot)
            if self.return_ll_only:
                return objective
            # Full objective - converted to bits per dimension
            b, c, h, w = x.shape
            bpd = (-objective) / (math.log(2.) * c * h * w)
            return z, bpd, y_logits

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape
        if self.use_binning_correction:
            x, logdet = uniform_binning_correction(x)
            raise
        else:
            logdet = torch.zeros(b).to(x.device)
        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        return z, objective, y_logits

    def reverse_flow(self, z, y_onehot, temperature, use_last_split=False, batch_size=0):
        if z is None:
            mean, logs = self.prior(z, y_onehot, batch_size=batch_size)
            z = gaussian_sample(mean, logs, temperature)
            self._last_z = z.clone()
        if use_last_split:
            for layer in self.flow.splits:
                layer.use_last = True
        x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True

    def logp(self, x):
        if self.return_ll_only:
            logp = self(x)
        else:
            logp = self(x)[0]
        return logp   

    def sample(self, bs):
        return self(y_onehot=None, temperature=1, batch_size=bs, reverse=True)

    def get_eval_samples(self, bs):
        with torch.no_grad():
            return self(y_onehot=None, temperature=1, batch_size=bs, reverse=True)
