# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
# sys.path: 这是一个Python列表，包含了解释器用于搜索模块的目录。当你尝试导入一个模块时，Python会在这个列表中查找该模块。
# Path: 这是pathlib模块中的一个类，用于表示文件系统路径并提供一系列方法和属性来操作这些路径。
# __file__: 这是一个内置变量，它表示当前被执行的Python文件的路径（仅当该文件作为主程序运行时才存在，如果是被导入的模块，则__file__是相对于导入它的脚本的位置）。
# Path(__file__): 创建一个Path对象，表示当前执行的Python文件的路径。
# .parent: Path对象的一个属性，表示当前路径的父目录。
# .absolute(): Path对象的一个方法，返回路径的绝对版本。
# .__str__(): 这是任何Python对象都有的方法，用于获取该对象的字符串表示。但在这种情况下，其实可以直接使用str()函数或简单地调用对象（因为Path对象定义了__str__和__repr__方法，使得你可以直接打印或使用str()来转换它）。
# sys.path.append(...): 将这个字符串表示的目录路径添加到sys.path列表中，这样Python在导入模块时就会在这个目录中查找。
# 这行代码的目的是将当前Python文件的祖父目录的绝对路径添加到Python的模块搜索路径中。这通常在你想要从不在当前工作目录或其子目录中的位置导入模块时很有用。
sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
#获取一个日志记录器（logger）对象。这个对象用于在你的代码中记录（或“日志”）消息。
#的__name__是一个特殊的内置变量，它表示当前模块的名字。例如，如果你在一个名为my_module.py的文件中执行这行代码，那么__name__的值就会是'my_module'。
#由于每个模块都有自己的名字，因此每个模块都可以有自己的日志记录器。这使得你可以轻松地跟踪来自不同模块的日志消息。
#使用__name__作为日志记录器的名字减少了在不同模块中意外使用相同名称的日志记录器的可能性。
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
#from mmcv.ops import DeformConv2dPack as DCN

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    # + `nc`（默认值为80）：类的数量，即要检测的目标类别数。
	# + `anchors`（默认值为空元组）：锚点（anchor）的列表或数组。锚点是预定义的边界框形状和大小，用于在目标检测中作为参考。
	# + `ch`（默认值为空元组）：每个检测层的输入通道数列表。
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()  #继承父类的初始化方法
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  #将锚点转换为PyTorch张量，并重新调整其形状以匹配检测层和锚点的数量。
        self.register_buffer('anchors', a)  # shape(nl,na,2)    #将锚点张量注册为缓冲区。缓冲区是模型的一部分，但不参与梯度计算。
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)。为锚点创建一个网格表示，并注册为缓冲区。这个网格表示可能用于后续的卷积或上采样操作。
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv。使用 `nn.ModuleList` 创建一个卷积层列表。`nn.ModuleList` 是一个包含其他模块的列表，允许其中的模块被正确地注册和追踪。
        #self.m = nn.ModuleList(DCN(x, self.no * self.na, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1) for x in ch)  # output DCN conv3x3

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        logits_ = []
        self.training |= self.export   #如果 self.export 是 True，则将 self.training 设置为 True（无论它之前的值是什么）。如果 self.export 是 False，则 self.training 的值保持不变。
        for i in range(self.nl):   #self.nl 是检测层的数量
            x[i] = self.m[i](x[i])  # conv，对于每个检测层 i，x[i] 是该层的输入特征图。self.m[i] 是一个卷积层（从前面的代码中我们知道它是nn.Conv2d或类似的层），它对该特征图进行卷积操作，并产生新的特征图。
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)，获取了卷积后特征图的形状。其中，bs 是批量大小（batch size），_ 是一个不需要的维度（通常是特征图的通道数），ny 和 nx 分别是特征图的高和宽。
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 使用 view 方法重塑特征图，以适应YOLO的检测输出格式。具体来说：
            # bs: 批量大小
            # self.na: 每个检测层的锚点数量
            # self.no: 每个锚点的输出数量（通常包括边界框的坐标、置信度和类别预测）
            # ny 和 nx: 特征图的高和宽
            # 由于PyTorch的view方法后的维度顺序是固定的（批量大小、通道数、高、宽），
            # 需要先使用view将其重塑为(bs, self.na, self.no, ny, nx)，
            # 然后使用permute将其重新排列为(bs, self.na, ny, nx, self.no)。
            # .contiguous()确保张量是内存连续的，这对于后续操作（如再次使用view或其他需要连续内存的操作）是必要的。

            if not self.training:  # inference
                # 它检查当前层的网格（self.grid[i]）的高和宽是否与当前特征图（x[i]）的高和宽匹配。
                # 如果不匹配，它会调用一个名为_make_grid的方法来创建一个新的网格，并将其移动到与特征图相同的设备上。
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                logits = x[i][..., 5:]    #logits是从特征图x[i]`中提取的，它对应于除了边界框坐标和置信度之外的预测（例如，类别预测）。

                #首先，对整个特征图应用sigmoid激活函数。然后，对边界框的坐标（xy）和大小（wh）进行调整
                #坐标（`xy`）通过网格（`self.grid[i]`）和步长（`self.stride[i]`）进行调整。
                #大小（`wh`）通过锚点网格（`self.anchor_grid[i]`）进行调整，并应用了平方操作。
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))   #将调整后的特征图y和logits重塑为适当的形状，并分别添加到z和logits_列表中。
                #logits_ 是一个列表，用于存储经过重塑（view）后的 logits
                logits_.append(logits.view(bs, -1, self.no - 5))   #在目标检测模型中，尤其是在YOLO（You Only Look Once）系列中，logits 通常指的是模型最后一层输出的原始预测值，这些值还没有经过任何激活函数或后处理。对于YOLO这样的单阶段检测器，这些 logits 通常包含边界框的位置、大小、置信度以及类别概率。

        return x if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    #@staticmethod装饰器用于指示该方法是一个静态方法。静态方法与类本身相关联，但不接受self（实例对象）或cls（类对象）作为第一个参数。
    # 它们通常用于实现与类本身相关但不需要访问或修改类状态的功能。
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    # 假设 xv 和 yv 都是形状为 (batch_size, height, width) 的张量，其中 batch_size 是批量中的样本数量，height 和 width 是特征图的高度和宽度。
    # 这两个张量可能分别包含特征图上的x坐标和y坐标网格。
    # 使用 torch.stack((xv, yv), 2) 将沿着第三个维度（即通道维度）堆叠这两个张量，结果将是一个形状为 (batch_size, height, width, 2) 的张量，其中最后一个维度的大小为2，对应于x和y坐标。


class Model(nn.Module):

    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):   # 检查一个对象（这里是cfg）是否是给定类型（这里是dict）的实例
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        #从self.yaml字典中获取'ch'键的值。
        #如果'ch'键存在，则将其值赋给self.yaml['ch']和ch。
        #如果'ch'键不存在，则将ch（这行代码之前的值）赋给self.yaml['ch']（在字典中添加新键值对）和ch。
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
            #deepcopy 函数创建一个 self.yaml 的深拷贝。深拷贝意味着它会复制 self.yaml 对象及其所有子对象，确保原始对象和复制对象是完全独立的，修改其中一个不会影响另一个。
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist，按照cfg生成模型
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()  .m:最后一个模型实例
        # print(m)

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])

            #当你想根据特征图的步长或下采样率来调整锚点的大小时。通过这样做，你可以确保锚点在原始输入图像上的大小是合适的。
            m.anchors /= m.stride.view(-1, 1, 1)   #m.stride.view(-1, 1, 1): 这里使用了view方法来改变m.stride张量的形状。-1是一个特殊值，它告诉PyTorch自动计算该维度的大小，以便保持张量中的元素总数不变。1, 1指定了其他两个维度的大小。因此，这个操作将m.stride转换为一个三维张量，其中第一个维度的大小是自动计算的，而后两个维度的大小都是1。
            #m.anchors /= ...: 这是一个逐元素除法操作。它将m.anchors中的每个元素都除以m.stride.view(-1, 1, 1)中对应位置的元素。由于m.stride.view(-1, 1, 1)的后两个维度都是1，所以这个除法操作实际上是将m.anchors的每个二维平面（对应于m.stride的一个元素）上的所有元素都除以同一个值。

            check_anchor_order(m)
            self.stride = m.stride
            #self._initialize_biases()  # only run once

        # Init weights, biases
        #调用initialize_weights来初始化模型的权重，然后调用info方法来输出模型的信息，最后使用日志记录器记录一条信息，表明模型已经准备好进行训练。
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width。在PyTorch中，一个批次的图像通常是 [B, C, H, W]，其中 B 是批次大小，C 是通道数，H 是高度，W 是宽度）
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  #flip 表示翻转操作
                yi = self.forward_once(xi)[0]  # forward。 [0] 表示我们只取返回值的第一个元素
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud，如果图像被垂直翻转（fi == 2），则对 y 坐标进行去翻转操作（即，用图像高度减去 y 坐标）
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr，如果图像被水平翻转（fi == 3），则对 x 坐标进行去翻转操作（即，用图像宽度减去 x 坐标）
                y.append(yi)
            #None：这是返回值的第二部分，它是一个None对象。在某些情况下，这可能是一个占位符或用于指示某种条件（比如没有额外的输出或信息）。
            return torch.cat(y, 1), None  # augmented inference, train。 torch.cat(y, 1)：这会将列表y中的所有张量在第1个维度（索引为1的维度）上进行连接。这要求列表y中的所有张量在其他维度上的大小都是相同的，除了要在其上连接的维度（即第1个维度）。
            #假设y是一个包含三个张量的列表，每个张量的形状都是(batch_size, C, H, W)（即批次大小、通道数、高度和宽度），那么torch.cat(y, 1)的结果将是一个形状为(batch_size, 3*C, H, W)的张量。
        else:
            return self.forward_once(x, x2, profile)  # single-scale inference, train


    def forward_once(self, x, x2, profile=False):
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                #nputs=(x,) 定义了模型的输入。注意这里是一个元组，即使你只有一个输入x，也需要用逗号将其变成元组形式，否则它会被解释为位置参数而不是元组。
                #verbose=False 表示在输出时不显示详细信息，只返回结果。
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):   #空循环10次，m(x)执行10次
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')   #dt的最后一个元素、o的值、m.np的值（四舍五入到整数）以及m.type的值，所有这些值都按照指定的宽度和对齐方式进行格式化。            if m.f == -4:
            if m.f == -4:
                x = m(x2)  #ir
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            i += 1

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)。   -1是一个特殊的值，表示该维度的大小是自动计算的，以便满足张量的总元素数量不变。换句话说，如果张量的总元素数量是已知的（在这里是mi.bias的元素数量），并且你已经为其他维度指定了大小（在这里是m.na），那么-1所代表的维度的大小就是自动计算出来的。
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            #将b（可能是从mi.bias转换而来的）重新塑形为一维张量，然后将其包装为一个Parameter对象，并指定该参数在训练时需要计算梯度。
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)    #view(-1)实际上是将b（如果它之前是多维的）转换回一维形式。
            #将一个张量包装在Parameter中时，这个张量就会被视为模型的一个参数，这意味着PyTorch的优化器将会知道在训练时更新这个张量。


    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            #对一个张量调用.detach()时，得到的是该张量的一个新的副本，但这个副本与计算图是分离的。这意味着，如果对原始张量（在这个例子中是mi.bias）进行任何后续操作或更新，这个新的、分离的副本（即b）将不会受到这些更改的影响。

            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS。检查模型中的最后一层是否是NMS类型的，并将结果存储在present变量中。
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()   #将模型设置为评估模式（evaluation mode）
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove，删除模型的最后一层。self.model[:-1] 使用了Python的切片语法来获取self.model中除了最后一个元素之外的所有元素。
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors。anchors[0] 是 [10, 13, 16, 30, 33, 23]，这个列表的长度是 6。因此，na = 6 // 2 = 3。这意味着 anchors[0] 包含了 3 个 anchor boxes。
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    #5 个参数用于边界框回归，分别是 (tx, ty, tw, th, to)（或类似的表示），其中 (tx, ty) 是边界框中心点的偏移量，(tw, th) 是边界框的宽度和高度的缩放因子，而 to（在某些实现中可能不存在）可能是一个额外的参数，用于表示目标的存在性（即该 anchor box 是否包含目标）。
    #no = na * (nc + 5) 表示在一个空间位置（一个网格单元或一个像素点）上，模型需要预测的总输出数量，这个数量是 anchor boxes 的数量乘以每个 anchor box 需要预测的参数数量。

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args。用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标
        m = eval(m) if isinstance(m, str) else m  # eval strings。eval() 是一个内置函数，它接受一个字符串作为参数，并尝试执行这个字符串作为Python代码。
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)   #确保某个值（在这里是c2 * gw）是某个除数（在这里是8）的整数倍
                args = [c1, c2, *args[1:]]
            elif m is Conv and args[0] == 64:    # new
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats。 args 列表的第3个位置（索引从0开始计数）插入元素 n
                    n = 1

        elif m is ResNetlayer:
            if args[3] == True:
                c2 = args[1]
            else:
                c2 = args[1]*4
        elif m is VGGblock:
            c2 = args[2]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Add, DMAF]:
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is NiNfusion:
            c1 = sum([ch[x] for x in f])
            c2 = c1 // 2
            args = [c1, c2, *args]
        elif m is TransformerFusionBlock:
            c2 = ch[f[0]]
            args = [c2, *args[1:]]
        else:
            c2 = ch[f]

        #在m(*args)中，*args会将args列表或元组中的元素解包，并作为独立的参数传递给m。例如，如果args = [1, 2, 3]，那么m(*args)相当于m(1, 2, 3)。
        #使用*操作符来解包上面创建的列表，并将其元素（即m的多个实例）传递给nn.Sequential。
        #如果n大于1，它会创建一个包含n个m实例的nn.Sequential容器，并将结果赋值给m_。如果n不大于1（即n等于1或0），它只会调用一次m(*args)并将结果赋值给m_。
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module。创建一个列表，其中每个元素都是通过调用m(*args)得到的。_是一个常用的占位符，表示我们在这个循环中不关心索引值。这个列表推导式会重复调用m(*args) n次。
        t = str(m)[8:-2].replace('__main__.', '')  # module type。将子串"__main__."替换为空字符串（即删除它）。
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []

        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/fqy/proj/paper/YOLOFusion/models/transformer/yolov5s_fusion_transformer(x3)_vedai.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)


    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    input_ir = torch.Tensor(8, 3, 640, 640).to(device)

    output = model(input_rgb, input_ir)
    
