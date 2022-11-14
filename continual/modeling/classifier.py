from torch import nn
import torch
from torch.nn import functional as F
from detectron2.engine.hooks import HookBase


class WA_Hook(HookBase):
    def __init__(self, model, step, distributed=True):
        self.classifier = None
        if distributed:
            self.classifier = model.module.sem_seg_head.predictor.class_embed.cls
        else:
            self.classifier = model.sem_seg_head.predictor.class_embed.cls
        self.step = step
        self.iteration = 0

    def after_step(self):
        if self.trainer.iter % self.step == 0:
            with torch.no_grad():
                new_cls = self.classifier[-1].weight
                old_cls = torch.cat([c.weight for c in self.classifier[1:-1]], dim=0)
                norm_new = torch.norm(new_cls, dim=1)
                norm_old = torch.norm(old_cls, dim=1)
                gamma = torch.mean(norm_old) / torch.mean(norm_new)
                self.classifier[-1].weight.mul_(gamma)


class IncrementalClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256, bias=True):
        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=bias) for c in classes])
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
        self.cls = nn.ModuleList(
            [nn.Linear(channels, c, bias=False) for c in classes])
        self.norm_feat = norm_feat
        self.scaler = scaler

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls[1:]:
            out.append(self.scaler * F.linear(x, F.normalize(mod.weight, dim=1, p=2)))
        out.append(self.scaler * F.linear(x, F.normalize(self.cls[0].weight, dim=1, p=2)))  # put as last the void class
        return torch.cat(out, dim=2)
