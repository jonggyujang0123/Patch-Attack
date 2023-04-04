from utils.base_utils import WarmupLinearSchedule
from models.lenet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F



class student():
    def __init__(
            self,
            warmup_steps,
            t_total,
            net = 'LeNet',
            n_student = 4,
            lr = 3e-2,
            ):
        self.st_loss = nn.CrossEntropyLoss()
        self.n_student = n_student
        if net=='LeNet':
            self.model = [LeNet() for i in range(n_student)]
        self.optim = [torch.optim.SGD(self.model[i].parameters(), lr = lr) for i in range(n_student)]
        self.scheduler = [WarmupLinearSchedule(self.optim[i], warmup_steps = warmup_steps, t_total = t_total) for i in range(n_student)]

    def train(self):
        for i in range(self.n_student):
            self.model[i].train()
    
    def eval(self):
        for i in range(self.n_student):
            self.model[i].eval()

    def to(self, dev):
        for i in range(self.n_student):
            self.model[i].to(dev)
        return self

    def update(self, x, y):
        self.train()
        bs = x.shape[0]
        idx = F.one_hot(torch.multinomial(torch.ones(self.n_student), bs, replacement=True)).unsqueeze(2).to(x.device)
        for i in range(self.n_student):
            self.optim[i].zero_grad()
        outputs = [self.model[i](x).unsqueeze(1) for i in range(self.n_student)]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.sum(outputs * idx, dim=1)
        loss = self.st_loss(outputs, y)
        if True:
            loss.backward()
    
            for i in range(self.n_student):
                self.optim[i].step()
                self.scheduler[i].step()
            
        return loss

    def OH_loss(self, logits):
        """
        input dimension : (bs, n_student, n_classes)
        """
        pseudo_label = logits.max(dim=2)[1]
        return self.st_loss(logits.reshape([-1, logits.shape[2]]), pseudo_label.detach().reshape([-1]))

    def forward(self, x):
        self.eval()
        outputs = [self.model[i](x).unsqueeze(1) for i in range(self.n_student)]
        outputs = torch.cat(outputs, dim=1)
        oh_loss = self.OH_loss(outputs)
#        outputs = outputs.softmax(dim=2).mean(dim=1)
        outputs = outputs.mean(dim=1).softmax(dim=1)
        return outputs, oh_loss


        
        


