# -----------------------------
# lite_detector.py
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 기본 모듈
# -----------------------------
class DWConv(nn.Module):
    """Depthwise + Pointwise Convolution"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class Residual(nn.Module):
    """Residual block with 2 DWConv"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DWConv(ch, ch)
        self.conv2 = DWConv(ch, ch)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# -----------------------------
# Backbone
# -----------------------------
class LiteBackbone(nn.Module):
    def __init__(self, in_ch=3, widths=[32,64,128,256]):
        super().__init__()
        self.stem = DWConv(in_ch, widths[0], 3, 2, 1)
        self.stage1 = nn.Sequential(DWConv(widths[0], widths[1], 3, 2, 1), Residual(widths[1]))
        self.stage2 = nn.Sequential(DWConv(widths[1], widths[2], 3, 2, 1), Residual(widths[2]))
        self.stage3 = nn.Sequential(DWConv(widths[2], widths[3], 3, 2, 1), Residual(widths[3]))

    def forward(self, x):
        p1 = self.stage1(self.stem(x))
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        return [p1, p2, p3]

# -----------------------------
# FPN
# -----------------------------
class LiteFPN(nn.Module):
    def __init__(self, in_chs=[64,128,256], out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([DWConv(c, out_ch, 1, 1, 0) for c in in_chs])
        self.smooth = nn.ModuleList([DWConv(out_ch, out_ch, 3, 1, 1) for _ in in_chs])

    def forward(self, feats):
        feats = [l(f) for l, f in zip(self.lateral, feats)]
        td2 = feats[2]
        td1 = feats[1] + F.interpolate(td2, size=feats[1].shape[2:], mode='nearest')
        td0 = feats[0] + F.interpolate(td1, size=feats[0].shape[2:], mode='nearest')
        outs = [s(td) for s, td in zip(self.smooth, [td0, td1, td2])]
        return outs

# -----------------------------
# Detection Head
# -----------------------------
class Head(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        self.obj = nn.Conv2d(in_ch, 1, 1)
        self.reg = nn.Conv2d(in_ch, 4, 1)

    def forward(self, feats):
        obj_preds = [self.obj(f) for f in feats]
        reg_preds = [self.reg(f) for f in feats]
        return obj_preds, reg_preds

# -----------------------------
# LiteDetector
# -----------------------------
class LiteDetector(nn.Module):
    """Anchor-free lightweight detector"""
    def __init__(self):
        super().__init__()
        self.backbone = LiteBackbone()
        self.neck = LiteFPN()
        self.head = Head()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        obj_preds, reg_preds = self.head(feats)
        return obj_preds, reg_preds
