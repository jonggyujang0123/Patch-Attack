from utils.base_utils import WarmupLinearSchedule
from models.lenet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
class student(nn.Module):
    def __init__(
            self,
            warmup_steps,
            t_total,
            net = 'LeNet',
            n_student = 4,
            lr = 1e-2,
            ):
        super().__init__()
        self.n_student = 4
        if net=='LeNet':
            self.model = [LeNet() for i in range(n_student)]
        self.optim = [torch.optim.SGD(self.model[i].parameters(), lr = lr) for i in range(n_student)]
        self.scheduler = [WarmupLinearSchedule(self.optim[i], warmup_steps = warmup_steps, t_total = t_total) for i in range(n_student)]
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for i in range(self.n_student):
            self.model[i].train()
    
    def eval(self):
        for i in range(self.n_student):
            self.model[i].eval()

    def update(self, x, y):
        self.train()
        bs = x.shape[0]
        idx = F.one_hot(torch.multinomial(torch.ones(self.n_student), bs, replacement=True)).unsqueeze(2).to(x.device)
        for i in range(self.n_student):
            self.optim[i].zero_grad()
        outputs = [self.model[i](x).unsqueeze(1) for i in range(self.n_student)]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.sum(outputs * idx, dim=1)
        loss = self.criterion(outputs, y)
        loss.backward()

        for i in range(self.n_student):
            self.optim[i].step()
            self.scheduler[0].step()
            
        return loss

    def forward(self, x):
        self.eval()
        outputs = [self.model[i](x).unsqueeze(1) for i in range(self.n_student)]
        outputs = torch.cat(outputs, dim=1).mean(dim=1)
        outputs = F.softmax(outputs)
        return outputs


        
        


