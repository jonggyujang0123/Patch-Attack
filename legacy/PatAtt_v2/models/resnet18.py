"""
Mnist tutorial main model
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class R_18_MNIST(nn.Module):
    def __init__(self, num_classes:int=3, num_channel:int=3):
        super().__init__()
        self.model = resnet18(pretrained= True)
        if num_channel==1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features,in_features)
        self.cls_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_classes))


    def forward(self, x):
        x = self.model(x)
        x = self.cls_head(x)
        return x
