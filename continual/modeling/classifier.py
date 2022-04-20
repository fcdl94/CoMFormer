from torch import nn
import torch
from torch.nn import functional as F


class IncrementalClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256):
        super().__init__()
        classes = [1] + classes
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=True) for c in classes])
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=3)
        out = []
        for mod in self.cls[1:]:
            out.append(mod(x))
        out.append(self.cls[0](x))  # put as last the void class
        return torch.cat(out, dim=2)


class CosineClassifier(nn.Module):
    def __init__(self, classes, norm_feat=True, channels=256, scaler=10.):
        super().__init__()
        self.cls = nn.ModuleList([nn.Linear(channels, 1)] +
            [nn.Linear(channels, c, bias=False) for c in classes])
        self.norm_feat = norm_feat
        self.scaler = scaler

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls[1:]:
            out.append(self.scaler * F.linear(x, F.normalize(mod.weight, dim=1, p=2)))
        out.append(self.cls[0](x))  # put as last the void class
        return torch.cat(out, dim=2)
