# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath

from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion*c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion*c2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=self.expansion*c2, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*c2),
                                          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(c2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion*c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y


class DMAF(nn.Module):
    def __init__(self, c2):
        super(DMAF, self).__init__()


    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    #DMFF
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        # for loop in range(self.loops):
        #     # with Learnable Coefficient
        #     rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
        #     rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
        #     ir_att_out  = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
        #     rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
        #     ir_fea_flat  = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))


        # with Learnable Coefficient
        rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
        rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
        ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
        rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
        ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]


class PositionAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(PositionAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        dwave_fea_flat = x   #rgb_fea_flat + ir_fea_flat

        b_s, nq = dwave_fea_flat.shape[:2]
        nk = dwave_fea_flat.shape[1]

        # Self-Attention
        dwave_fea_flat = self.LN1(dwave_fea_flat)
        q_dwave = self.que_proj(dwave_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_dwave  = self.key_proj(dwave_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_dwave  = self.val_proj(dwave_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_dwave = torch.matmul(q_dwave, k_dwave) / np.sqrt(self.d_k)

        # get attention matrix
        att_dwave = torch.softmax(att_dwave, -1)
        att_dwave = self.attn_drop(att_dwave)

        # output
        out_dwave = torch.matmul(att_dwave, v_dwave).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_dwave = self.resid_drop(self.out_proj(out_dwave)) # (b_s, nq, d_model)

        return out_dwave


class PositionTransformerBlock(nn.Module):
    #DMFF
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(PositionTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.paatt = PositionAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)

        self.mlp_dwave = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()

    def forward(self, x):
        # with Learnable Coefficient
        dwave_fea_out = self.paatt(x)
        new_att_out = self.coefficient1(x) + self.coefficient2(dwave_fea_out)
        new_fea_flat = self.coefficient3(new_att_out) + self.coefficient4(self.mlp_dwave(self.LN1(new_att_out)))

        return new_fea_flat

#
# class ScaledDotProductAttention(nn.Module):
#     '''
#     Scaled dot-product attention
#     '''
#
#     def __init__(self, d_model, h, kernel_size=3, dropout=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(ScaledDotProductAttention, self).__init__()
#
#         self.d_model = d_model
#         self.d_k = d_model // h
#         self.d_v = d_model // h
#         self.h = h
#
#         self.conv_q = nn.Conv2d(self.d_model, self.h * self.d_k, kernel_size, padding=(kernel_size - 1) // 2)
#         self.conv_k = nn.Conv2d(self.d_model, self.h * self.d_k, kernel_size, padding=(kernel_size - 1) // 2)
#         self.conv_v = nn.Conv2d(self.d_model, self.h * self.d_v, kernel_size, padding=(kernel_size - 1) // 2)
#         self.gamma_o = nn.Parameter(torch.zeros(1))
#
#         self.dropout=nn.Dropout(dropout)
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes
#         :param queries: Queries (b_s, nq, d_model)
#         :param keys: Keys (b_s, nk, d_model)
#         :param values: Values (b_s, nk, d_model)
#         :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#         :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         :return:
#         '''
#         b_s, ch, height, width = x.size()
#
#         q = self.conv_q(x).view(b_s, -1, height * width).permute(0, 2, 1)
#         k = self.conv_k(x).view(b_s, -1, height * width)
#         v = self.conv_v(x).view(b_s, -1, height * width).permute(0, 2, 1)
#
#         att = torch.matmul(q, k) / np.sqrt(self.d_k)
#         if attention_weights is not None:
#             att = att * attention_weights
#         if attention_mask is not None:
#             att = att.masked_fill(attention_mask, -np.inf)
#         att = torch.softmax(att, -1)
#         att = self.dropout(att)
#
#         out = torch.matmul(att, v).view(b_s, ch, height, width)
#         out = self.gamma_o * out + x
#         return out
#
# class PositionAttentionModule(nn.Module):
#     def __init__(self, d_model, head=1, kernel_size=3):
#         super().__init__()
#         self.cnn1 = nn.Conv2d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2)
#         self.pa   = ScaledDotProductAttention(d_model, head, kernel_size)
#         self.cnn2 = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
#                                   nn.BatchNorm2d(d_model),
#                                   nn.ReLU())
#         self.cnn3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(d_model, d_model, 1))
#
#     def forward(self, x):
#         feat      = self.cnn1(x)
#         feat_pa   = self.pa(feat)
#         feat_conv = self.cnn2(feat_pa)
#         new_feat  = self.cnn3(feat_conv)
#
#         return new_feat
#
#
# class SimplifiedScaledDotProductAttention(nn.Module):
#     '''
#     Scaled dot-product attention
#     '''
#
#     def __init__(self, d_model, h, dropout=.1):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         '''
#         super(SimplifiedScaledDotProductAttention, self).__init__()
#
#         self.d_model = d_model
#         self.d_k = d_model//h
#         self.d_v = d_model//h
#         self.h = h
#
#         self.init_weights()
#
#         self.gamma_o = nn.Parameter(torch.zeros(1))
#
#         self.dropout = nn.Dropout(dropout)
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x, attention_mask=None, attention_weights=None):
#         '''
#         Computes
#         :param queries: Queries (b_s, nq, d_model)
#         :param keys: Keys (b_s, nk, d_model)
#         :param values: Values (b_s, nk, d_model)
#         :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#         :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         :return:
#         '''
#         b_s, ch, height, width = x.size()  # queries.shape[:2]
#
#         q = x.view(b_s, ch, -1)  # (b_s, nq, H*W)
#         k = x.view(b_s, ch, -1).permute(0, 2, 1)  # (b_s, H*W, nk)
#         v = x.view(b_s, ch, -1)  # (b_s, nk, , H*W)
#
#         att = torch.matmul(q, k) / np.sqrt(self.d_k)
#
#         if attention_weights is not None:
#             att = att * attention_weights
#         if attention_mask is not None:
#             att = att.masked_fill(attention_mask, -np.inf)
#         att = torch.softmax(att, -1)
#         att = self.dropout(att)
#
#         out = torch.matmul(att, v).view(b_s, ch, height, width)
#         out = self.gamma_o * out + x
#
#         return out
#
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, d_model, head=1, kernel_size=3):
#         super().__init__()
#         self.cnn1  = nn.Conv2d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2)
#         self.ca    = SimplifiedScaledDotProductAttention(d_model, head)
#
#         self.cnn2 = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
#                                   nn.BatchNorm2d(d_model),
#                                   nn.ReLU())
#         self.cnn3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(d_model, d_model, 1))
#
#     def forward(self, x):
#         feat     = self.cnn1(x)
#         feat_ca  = self.ca(feat)
#         feat_conv= self.cnn2(feat_ca)
#         new_feat = self.cnn3(feat_conv)
#
#         return new_feat

# class ChannelAttentionModule_posatt_crossatt(nn.Module):
#     def __init__(self, d_model,vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
#                  attn_pdrop=0.1, resid_pdrop=0.1):
#         super().__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#         self.pos_emb_ir  = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
#         self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')
#
#         # LearnableCoefficient
#         self.vis_coefficient = LearnableWeights()
#         self.ir_coefficient = LearnableWeights()
#
#         # init weights
#         # self.apply(self._init_weights)
#
#         # cross transformer
#         self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])
#         # self.positiontransformer = PositionTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
#
#         # self.gamma_ir = nn.Parameter(torch.zeros(1))
#         # self.gamma_rgb = nn.Parameter(torch.zeros(1))
#         # Concat
#         # self.concat = Concat(dimension=1)
#
#         # conv1x1
#         # self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, x):
#         rgb_fea = x[0]
#         ir_fea  = x[1]
#         assert rgb_fea.shape[0] == ir_fea.shape[0]
#         bs, c, h, w = rgb_fea.shape
#
#         # ------------------------- cross-modal feature fusion -----------------------#
#         #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
#         #SFS
#         new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
#         new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
#         rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
#         #new_rgb_fea.contiguous(): 确保new_rgb_fea是连续的，以便后续操作可以安全地执行。
#         #.view(bs, new_c, -1): 改变new_rgb_fea的形状。假设原始形状是(bs, H, W, new_c)（其中H和W是高和宽），这个操作会将其重塑为(bs, new_c, H*W)。注意，-1会被自动计算为H*W。
#         #.permute(0, 2, 1): 将张量的形状从(bs, new_c, H*W)更改为(bs, H*W, new_c)。
#         #+ self.pos_emb_vis: 将结果与self.pos_emb_vis相加。假设self.pos_emb_vis的形状是(1, H*W, new_c)或(bs, H*W, new_c)，那么这两个张量就可以相加（前提是它们的形状是兼容的）。
#         #最终，rgb_fea_flat将是一个形状为(bs, H*W, new_c)的张量，它是new_rgb_fea经过重塑、维度交换和与位置嵌入相加后得到的。
#
#         #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
#         new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
#         ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
#
#         rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
#
#         # dwave_fea_flat = self.positiontransformer([rgb_fea_flat, ir_fea_flat])
#
#         rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
#         new_rgb_fea = rgb_fea_CFE + rgb_fea
#
#         ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         if self.training == True:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
#         else:
#             ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
#         new_ir_fea = ir_fea_CFE + ir_fea
#
#         # dwave_fea_CFE = dwave_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
#         # if self.training == True:
#         #     dwave_fea_CFE = F.interpolate(dwave_fea_CFE, size=([h, w]), mode='nearest')
#         # else:
#         #     dwave_fea_CFE = F.interpolate(dwave_fea_CFE, size=([h, w]), mode='bilinear')
#         # new_dwave_fea = dwave_fea_CFE
#
#         # new_fea = self.concat([new_rgb_fea, new_ir_fea])  #ir+rgb fusion
#         # new_fea = self.conv1x1_out(new_fea)
#         #
#         # return new_fea
#         # return new_rgb_fea + new_ir_fea
#         return [new_rgb_fea, new_ir_fea]

class DoubleAttentionModule(nn.Module):
    def __init__(self, d_model,vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir  = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()
        self.dwave_coefficient = LearnableWeights()

        # cross transformer
        self.crosstransformer = CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
        self.positiontransformer = PositionTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea  = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #SFS
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat_in = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat_in = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat_in, ir_fea_flat_in])

        # dwave_fea_flat = rgb_fea_flat + ir_fea_flat
        dwave_fea_flat_in = self.dwave_coefficient(rgb_fea_flat_in, ir_fea_flat_in)
        dwave_fea_flat = self.positiontransformer(dwave_fea_flat_in)

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea

        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        dwave_fea_CFE = dwave_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            dwave_fea_CFE = F.interpolate(dwave_fea_CFE, size=([h, w]), mode='nearest')
        else:
            dwave_fea_CFE = F.interpolate(dwave_fea_CFE, size=([h, w]), mode='bilinear')
        new_dwave_fea = dwave_fea_CFE

        return [new_rgb_fea, new_ir_fea, new_dwave_fea]


class DAModule_posatt_crossatt(nn.Module):

    def __init__(self, d_model, head=1, kernel_size=3):
        super().__init__()
        self.double_attention_module  = DoubleAttentionModule(d_model, head, kernel_size)
        self.concat = Concat(dimension=1)
        self.conv1x1_3 = Conv(c1=d_model * 3, c2=d_model, k=1, s=1, p=0, g=1, act=True)


    def forward(self, x):     #20240822,danetv2
        x_ca = x
        c_out = self.double_attention_module(x_ca)
        new_fea = self.concat([c_out[0], c_out[1], c_out[2]])  #ir+rgb fusion
        new_out = self.conv1x1_3(new_fea)

        return new_out

    # def forward(self, x):    #20240825,danetv2
    #     new_rgb_fea_ca = self.position_attention_module(x[0])
    #     new_ir_fea_ca  = self.position_attention_module(x[1])
    #     c_out = self.channel_attention_module([new_rgb_fea_ca, new_ir_fea_ca])
    #
    #     new_out = self.cnn(c_out[0] + c_out[1])
    #     return new_out

class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()
        # DAnet_patt_crossatt
        self.danet  = DAModule_posatt_crossatt(d_model, head=1, kernel_size=3)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]

        # 4.DAnet
        new_fea = [rgb_fea, ir_fea]
        new_fea = self.danet(new_fea)

        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
                                               nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
                                               )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out
