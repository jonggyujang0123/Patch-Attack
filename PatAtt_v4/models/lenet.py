from torch import nn
import torch.nn.functional as F
def variable_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class LeNet(nn.Module):
    def __init__(self,
            n_c = 1):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(n_c, 6, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(6),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(400, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84,10),
                )
        self.apply(variable_init)
    def forward(self, x):
        return self.net(x)


