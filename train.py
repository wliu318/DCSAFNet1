import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  #进度条

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo_test import Model   #采用yolo_test中的model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir
from utils.general import logger, labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from utils.datasets import RandomSampler
import global_var

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_rgb_ir(hyp, opt, device, tb_writer=None):
    os.environ["WANDB_MODE"] = "offline"
    #Weights & Biases 是一个用于机器学习实验的跟踪、可视化和调试的平台。通过设置 WANDB_MODE 环境变量，可以控制WandB的行为。
    #当 WANDB_MODE 被设置为 "offline" 时，WandB 将不会尝试与远程服务器通信，而是将所有数据保存在本地。这允许用户在不连接到互联网的情况下继续他们的实验，或者在没有网络连接的环境中工作。
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)  #将 Python 数据结构（如字典、列表等）序列化为 YAML 格式的字符串，并将这些数据写入到文件中
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(seed=1 + rank, deterministic=True)  #初始化随机数生成器的种子，以确保在多个进程或设备上的操作是可重复的或确定的。确保不同的进程或设备使用不同的种子，从而避免在并行计算中产生相同的随机数序列
                                                   #。当deterministic=True时，一些操作（如卷积、线性变换等）会使用固定的算法或实现，以确保在相同的输入和种子下总是产生相同的结果。这对于调试和比较不同实验的结果非常重要。
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = opt.data.endswith('coco.yaml')  #检查字符串是否以coco.yaml结束

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict 。创建了一个名为 loggers 的字典，其中包含一个键 'wandb'，其对应的值为 None。
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters

        # torch.load(weights) 将加载该文件中的对象，并返回该对象。如果该文件保存的是一个PyTorch模型或张量，那么torch.load()将返回该模型或张量。但是，如果该文件保存的是一个字典（例如，保存了模型权重和其他元数据的字典），那么torch.load()将返回这个字典。
        # get()用于从字典中获取wandb_id键的值。在WandB中，每次你开始一个新的实验或训练运行，它都会生成一个唯一的ID来标识这次运行。这个ID用于在WandB的UI中查找和比较不同的运行。
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)  #stem是pathlib.Path对象的一个属性，它返回路径中文件名（不包括扩展名）的部分
        loggers['wandb'] = wandb_logger.wandb  #'.wandb' 键通常用于标识 WandB 日志记录器，wandb_logger.wandb 可能是一个返回 WandB 日志记录器实例的函数或方法的调用结果，也可能是直接实例化的一个对象。这个对象将用于将信息发送到 WandB 平台。
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming


    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes,nc=3
    #names: ['person', 'car', 'bicycle']
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    #pretrained = False
    if pretrained:
        #attempt_download函数会检查本地是否存在该权重文件。如果不存在，它将尝试从指定的源下载文件。
        #如果在rank为0的进程上首次执行此操作，则其他进程将等待它完成，以确保所有进程都使用相同的权重文件。
        #在分布式训练环境中，确保只在特定的rank（可能是rank为0的进程）上首次下载模型权重文件，而其他进程将等待该文件被下载后再继续执行后续操作。这样可以避免多个进程同时下载同一文件可能导致的冲突或重复，并确保所有进程都使用相同的权重文件。
        with torch_distributed_zero_first(rank):  #当进入这个with语句块时，torch_distributed_zero_first(rank)将执行一些设置操作（可能是屏障操作，确保其他进程等待），并在退出该块时执行清理或通知其他进程的操作
            attempt_download(weights)  # download if not found locally
        #ckpt用于存储从 weights 文件中加载的数据
        #map_location=device 指定了数据应该被加载到哪个设备上
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        #opt.cfg：yolov5l_Transfusion_FLIR
        #或者 ckpt字典中'model'键对应的对象的yaml属性作为配置文件的路径

        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create. 超参文件中anchors被注释掉了
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32。从 ckpt 字典中加载模型参数（可能是整个模型实例或仅仅是模型的参数），将这些参数转换为单精度浮点数，然后提取这些参数的字典形式，并将其存储在 state_dict 变量中。
        #计算两个字典的交集。在深度学习的上下文中，这通常用于确保当从一个检查点（checkpoint）加载模型权重时，只加载与目标模型结构匹配的权重。
        #新的 state_dict（现在覆盖了原来的变量）将只包含与当前模型架构匹配的权重，并且排除了 exclude 列表中指定的任何权重。这个新的 state_dict 可以随后用于加载到模型实例中，以确保只加载与新模型架构兼容的权重。
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect, 对于公用模块，继承yolo5l的参数
        #对模型的state_dict（状态字典）中的键（即权重和偏置的名称）进行重命名。具体地，它修改了键的前缀部分，通常是为了与新的模型架构或命名约定保持一致
        new_state_dict = state_dict
        # for key in list(state_dict.keys()):    #20240824，liuwen
        #     #key[:6]：获取键的前6个字符。int(key[6]) + 10：将键的第7个字符（假设为数字）转换为整数，并加上10。key[7:]：获取键从第8个字符开始到末尾的所有字符。
        #     #将修改后的前缀、调整后的数字和原始键的其余部分连接起来，形成新的键名。
        #     #将键值付给新的键
        #     if key[6:8]=='10':   #第10个module的key6加10将变成110，module.110,????
        #         tmp = 1
        #     new_state_dict[key[:6] + str(int(key[6])+10) + key[7:]] = state_dict[key]   #因为是双流，第二路从十开始编号，所以加10，并复用了第一路的和模型参数,    #20240824，liuwen

        #从新的状态字典（new_state_dict）中加载模型参数
        #当strict设置为True（默认值）时，加载过程将更加严格：如果new_state_dict和模型的状态字典之间的键有任何不匹配，都会引发错误。
        #strict参数设置为False时：
        #如果new_state_dict中的键与模型（model）中的键完全匹配，则权重和偏置将被正常加载。
        #如果new_state_dict中存在模型（model）中没有的键，这些键将被忽略，并且不会引发错误。这允许你从具有更多参数的模型中加载权重，只要这些额外的参数不是模型当前架构所必需的。
        #如果模型（model）中存在new_state_dict中没有的键，这些键将不会被加载，但也不会引发错误。这通常意味着模型的某些部分（如新添加的层或参数）将不会被初始化，而是保持其默认初始化状态（通常是随机初始化）。
        model.load_state_dict(new_state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path_rgb = data_dict['train_rgb']
    test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    test_path_ir = data_dict['val_ir']
    labels_path = data_dict['path'] + '/labels/test'
    labels_list = os.listdir(labels_path)  #列出指定目录（labels_path）下的所有文件和子目录的名称，并将这些名称存储在labels_list这个列表中
    labels_list.sort()  #按照字母顺序排序

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)。freeze 列表是空的，这意味着 if any(x in k for x in freeze): 这个条件永远不会为真，因此不会有任何参数被冻结。
    #如果想冻结某些特定的参数，你需要往 freeze 列表中添加相应的参数名称（或名称中的部分字符串）。假设我们想冻结名为 'conv1.weight' 和 'conv2.bias' 的参数 ：freeze = ['conv1.weight', 'conv2.bias']
    for k, v in model.named_parameters():      #使用 model.named_parameters() 方法遍历模型的参数。这个方法会返回一个迭代器，每次迭代都会返回一个参数的名称（k）和对应的值（v）
        v.requires_grad = True  # train all layers。默认情况下所有的参数都会在反向传播时计算梯度，并可能在优化步骤中被更新。
        if any(x in k for x in freeze):   #检查当前参数的名称（k）是否包含 freeze 列表中的任何字符串。如果 k 包含 freeze 列表中的任何一个字符串，那么就会打印出一条消息
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size，名义批处理大小
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    #在某些情况下，为了模拟更大的批处理大小（即nbs），我们可能使用较小的实际批处理大小（即total_batch_size）并累积多个梯度步数（即accumulate）。然而，直接使用较小的批处理大小可能会导致正则化不足，因为权重更新的频率增加了。因此，通过增加weight_decay的值，我们可以补偿这种正则化不足的影响。
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay。weight_decay是一种正则化技术，用于防止模型过拟合。
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    #遍历一个PyTorch模型（model）的所有模块（modules）
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():   #k是模块的名称（包括其层次结构），v是模块本身。
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    # biases ,如果模块v有一个属性bias，并且这个bias是一个nn.Parameter（即它是可训练的），那么就将这个偏置添加到pg2参数组中。通常，偏置参数在训练时不需要应用权重衰减（或应用较小的权重衰减）。
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay .如果模块v是一个nn.BatchNorm2d（批量归一化层），那么就将它的权重添加到pg0参数组中。通常，批量归一化层的权重在训练时不应用权重衰减，因为它们是通过学习数据的缩放和偏移来工作的，而不是学习数据的特征表示。
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay.如果模块v有一个属性weight，并且这个weight是一个nn.Parameter（即它是可训练的），但v不是一个nn.BatchNorm2d（已经在前面的条件中处理过了），那么就将这个权重添加到pg1参数组中。这些权重在训练时通常会应用权重衰减。

    #代码只将pg0传递给了优化器，但在实际应用中，您可能还想优化其他参数组（如pg1和pg2）。
    # 为了同时优化多个参数组，您可以向优化器的构造函数传递一个包含所有参数组的列表，
    # 并为每个参数组指定不同的超参数（如果需要）。但是，这段代码仅展示了如何为pg0设置优化器。
    if opt.adam:
        #创建一个optim.Adam实例。它只优化pg0参数组（这些参数通常是不应用权重衰减的参数，如批量归一化层的权重）。
        # lr=hyp['lr0']设置了学习率，它来自hyp字典中的'lr0'键。
        # betas=(hyp['momentum'], 0.999)设置了Adam优化器的两个超参数β1和β2，
        # 其中β1被设置为hyp字典中的'momentum'值（注意这里的注释表明β1被调整以匹配动量值，
        # 但在标准的Adam中，β2通常是0.999，而β1通常是0.9），而β2被硬编码为0.999。
        #pg0,pg1,pg2,均为列表，优化器如何识别该列表的性质，或者说是什么参数？？？？？
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        #SGD优化器，它只优化pg0参数组。学习率、动量和Nesterov动量都被设置为hyp字典中的对应值。
        # Nesterov动量被启用（nesterov=True），它是一种改进的动量方法，可以提高在某些问题上的收敛性能。
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    #调用add_param_group传入一个字典，该字典描述了新的参数组的配置.
    # 'params'键的值是pg1，即要添加到优化器的参数组。
    # 'weight_decay'键的值是hyp['weight_decay']，是从超参数配置中获取的权重衰减值。
    # pg1中的参数将使用此指定的权重衰减值进行优化。
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    #调用add_param_group方法，只传入了一个包含'params'键的字典，其值为pg2。
    #没有为pg2指定额外的超参数（如权重衰减），因此它将使用优化器的默认设置（或之前为优化器设置的全局设置，如果有的话）进行优化。
    #pg2可能包含不应该应用权重衰减的参数，如偏置项。
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    #f-string字符串格式化方法
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(pg0)} weight, {len(pg1)} weight (no decay), {len(pg2)} bias")
    del pg0, pg1, pg2

    if opt.linear_lr:   #使用线性学习率调度策略
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear.
        # 定义了一个匿名函数（lambda函数）lf，它接受一个参数x（代表当前的训练周期或批次索引），并返回一个计算后的学习率值。
        #(1 - x / (epochs - 1))：这部分计算了一个从1递减到0的线性比例。其中，epochs是总的训练周期数，x是当前周期或批次的索引（假设从0开始）。
        #(1.0 - hyp['lrf'])：这里hyp['lrf']可能表示学习率衰减的最低值（即最终的学习率），所以(1.0 - hyp['lrf'])计算了从初始学习率到最低学习率的差值。
        #上述两部分相乘，并加上hyp['lrf']，得到了当前周期或批次的学习率值。这种调度方式会使得学习率从初始值线性递减到hyp['lrf']。
    else:               #使用余弦学习率调度策略
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf'].余弦学习率调度策略通常会在训练开始时设置一个较高的学习率，然后随着训练的进行，学习率会按照余弦函数的形式逐渐降低到一个较低的值（hyp['lrf']），并在接近训练结束时再次略微提高（这取决于one_cycle函数的具体实现）。

    #通过 lambda 函数动态地调整学习率。
    #lr_lambda=lf 指定了一个 lambda 函数 lf，这个函数用于计算新的学习率。
    # 这个 lambda 函数返回一个学习率乘数。这个乘数与优化器当前的学习率相乘，从而得到新的学习率。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA.模型指数移动平均（Exponential Moving Average, EMA）的一个条件赋值。
    # EMA用来平滑模型的权重更新，从而提高模型的稳定性和泛化能力。
    ema = ModelEMA(model) if rank in [-1, 0] else None
    #ModelEMA 类接受一个模型作为输入，并创建一个该模型的 EMA 副本。EMA 副本会跟踪原始模型的权重，并使用 EMA 来更新这些权重。
    #在分布式训练中，通常只需要一个进程（通常是主进程）来保存和更新 EMA 权重。
    # 其他进程（即非主进程）只负责训练模型并发送梯度更新到主进程。
    # 由于 EMA 只是原始模型权重的一个平滑版本，并且主要用于评估或推理，因此没有必要在每个进程上都保存 EMA 权重。
    # 这样可以节省内存和计算资源。


    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            #assert 检查 start_epoch 是否大于0。如果 start_epoch 不大于0（即小于或等于0），那么会触发断言错误，并显示一个错误消息，这个错误消息是关于模型训练的。
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            #当从检查点恢复训练时，知道模型已经被训练了多少轮，并据此确定还需要训练多少轮以达到你希望的总轮次。
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs
            #如果ckpt['epoch']是20（即模型已经被训练了20轮），而你原本打算微调5轮（即epochs为5），那么更新后的epochs将会是25，表示从检查点开始你还需要训练25轮。

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)？？？？？
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # print("nl", nl)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples.检查x（即图像大小）是否符合某种标准或限制gs

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader.创建两个对象：一个数据加载器（dataloader）和一个数据集（dataset）
    dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                                   hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                   world_size=opt.world_size, workers=opt.workers,
                                                   image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    #使用np.concatenate来连接dataset.labels中的多个数组（如果存在的话）。参数0表示在第一个轴（通常是行）上进行连接。
    #从一个二维数组中选择所有行（:表示选择所有行）和第一列（0表示选择第一列）
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir,imgsz_test, 1, gs, opt,
                                                        hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                                        rank=-1, world_size=opt.world_size, workers=opt.workers,
                                                        pad=0.5, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes.从（各种数据类型如列表、元组、NumPy数组等）labels创建张量（tensors）
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
            #model.half()将模型中的所有参数和缓冲区（buffers）从float32（或float64，如果它们原本就是double类型）转换为float16。
            #model.float()将模型中的所有参数和缓冲区从float16转换回float32

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    MRresult = 0.0
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        torch.cuda.empty_cache()
        model.train()  #将模型设置为训练模式

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)   #遍历一个可迭代对象（如列表、元组或字符串）并同时获取每个元素的索引和值
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'rank', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch ----i：批次索引，从 0 开始递增----数据加载器返回的每个批次（batch）包含四个元素：imgs（图像）、targets（标签或目标）、paths（图像路径，可能用于某些目的，如日志或调试）以及一个被忽略的元素（由下划线 _ 表示）。-----------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_rgb = imgs[:, :3, :, :]  #imgs 中选取了前三个通道
            imgs_ir = imgs[:, 3:, :, :]   #imgs 中选取了从第四个通道开始的所有通道

            #如果 imgs 的形状是 [64, 4, 224, 224]（表示一个批次中有 64 张 224x224 的四通道图像），那么：
            #imgs_rgb 的形状将是 [64, 3, 224, 224]
            #imgs_ir 的形状将是 [64, 1, 224, 224]（因为只选取了一个额外的通道）

            # FQY my code 训练数据可视化
            flage_visual = global_var.get_value('flag_visual_training_dataset')
            if flage_visual:
                from torchvision import transforms
                unloader = transforms.ToPILImage()   #将一个张量转换为PIL图像对象
                for num in range(batch_size):
                    #首先选择了RGB通道，然后将张量移动到CPU（如果它之前在GPU上），然后克隆了张量（以防止原地修改原始数据），
                    # 移除了假的批次维度（因为你只处理一个图像），然后将张量转换为PIL图像并保存。
                    #先是RGB图
                    image = imgs[num, :3, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)    #
                    image.save('example_%s_%s_%s_color.jpg'%(str(epoch), str(i), str(num)))
                    # 再是IR图
                    image = imgs[num, 3:, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_ir.jpg'%(str(epoch), str(i), str(num)))

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())  #interp一维线性插值
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            #amp.autocast是一个上下文管理器，它会自动地将此上下文内的所有Tensor操作转换为混合精度。
            # enabled=cuda意味着当CUDA（即GPU）可用时，启用混合精度。
            with amp.autocast(enabled=cuda):
                # pred = model(imgs)  # forward
                pred = model(imgs_rgb, imgs_ir)  # forward？？？？？？？？？为什么两个输入参数，没对应起来
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size。targets.to(device)确保目标值与预测值在同一个设备上（CPU或GPU）
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            #用于在反向传播之前对损失进行缩放。这通常与 PyTorch 的自动混合精度（Automatic Mixed Precision, AMP）库一起使用。
            #动态地缩放损失，以避免在更新权重时的下溢或上溢。在混合精度训练中，模型的前向传播通常使用半精度（通常是 float16）来加速计算，但由于 float16 的表示范围较小，直接对半精度梯度进行累积和更新可能会导致数值不稳定。
            #scaler.scale(loss): 这个方法将损失 loss 乘以一个缩放因子。这个缩放因子是 GradScaler 根据之前迭代的梯度更新情况动态计算的，以确保梯度更新不会过大或过小。通过这样做，混合精度训练可以在保持数值稳定性的同时，充分利用半精度计算的速度优势。
            #在调用 .backward() 之前，对损失进行缩放可以确保梯度计算是在一个合适的尺度上进行的。
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:    #accumulate 变量定义了累积多少个批次的梯度后再执行一次优化步骤
                scaler.step(optimizer)  # optimizer.step，在执行优化步骤之前对梯度进行缩放，以避免在更新权重时出现数值不稳定的问题。
                scaler.update()         # 更新缩放器的状态
                optimizer.zero_grad()   #每次权重更新之后，我们都需要清零优化器的梯度缓存。这是因为 PyTorch 默认会累积梯度，而不是在每个迭代中覆盖它们。如果不清零梯度，下一个迭代中的梯度将包括之前迭代的梯度，这会导致训练不稳定。
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                #获取当前PyTorch在CUDA（即GPU）上已预留的内存大小，并将其格式化为带有3位有效数字的字符串，单位为GB（G）。
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)     #已经创建了一个 tqdm 进度条对象 pbar，可以使用 set_description 方法来更改其描述或标题。s 是你想要设置的新描述或标题，它是一个字符串。

                if ni < 3 and epoch + 1 == epochs:    #创建了两个线程，每个线程都调用plot_images函数来保存图像
                    f1 = save_dir / f'train_batch{ni}_vis.jpg'
                    f2 = save_dir / f'train_batch{ni}_inf.jpg'
                    #daemon=True表示这些线程是守护线程。守护线程在主线程结束时会自动终止，无论它们的任务是否完成。
                    Thread(target=plot_images, args=(imgs_rgb, targets, paths, f1), daemon=True).start()
                    Thread(target=plot_images, args=(imgs_ir, targets, paths, f2), daemon=True).start()

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # 从PyTorch的优化器（optimizer）的param_groups中提取学习率
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])  #更新副本参数
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, MRresult, times = test.test(data_dict,
                                                           batch_size=1,
                                                           imgsz=imgsz_test,
                                                           model=ema.ema,
                                                           single_cls=opt.single_cls,
                                                           dataloader=testloader,
                                                           save_dir=save_dir,
                                                           save_txt=True,
                                                           save_conf=True,
                                                           verbose=nc < 50 and final_epoch,
                                                           plots=plots and final_epoch,
                                                           wandb_logger=wandb_logger,
                                                           compute_loss=compute_loss,
                                                           is_coco=is_coco,
                                                           labels_list=labels_list,
                                                           )   #最后一轮没有测试，开始测试

            # log
            keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/rank_loss',  # train loss
                    'TP', 'FP', 'FN', 'F1', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/rank_loss', 'val/ms-2',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2',  # learning rate
                    'MR_all', 'MR_day', 'MR_night', 'MR_near', 'MR_medium', 'MR_far', 'MR_none', 'MR_partial', 'MR_heavy', 'Recall_all'  # MR
                    ]
            vals = list(mloss) + list(results) + lr + MRresult   #list(*)：将 * 转换为列表（如果它本身不是列表的话）
            dicts = {k: v for k, v in zip(keys, vals)}  # dict。zip(keys, vals) 会将 keys 和 vals 中的元素配对成元组，然后字典推导式遍历这些元组，并将每个元组的第一个元素作为键（key），第二个元素作为值（value），来创建新的字典项。
            file = save_dir / 'results.csv'
            n = len(dicts) + 1  # number of cols
            #('%s,' * n % tuple(['epoch'] + keys)): 这部分代码首先创建了一个由n个'%s,'组成的字符串（例如，如果n=3，则字符串为'%s,%s,%s,'）。然后，它使用%操作符来格式化这个字符串，用tuple(['epoch'] + keys)作为参数。这意味着['epoch'] + keys列表会被转换为元组，并用于替换%s占位符。但是，由于字符串末尾有一个逗号，所以使用rstrip(',')来移除最后一个逗号。
            s = '' if file.exists() else (('%s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')
            # '%g,' * n：这会创建一个由 n 个 '%g,' 组成的字符串。
            # tuple([epoch] + vals)：这会创建一个元组，其中第一个元素是 epoch，后面跟着 vals 列表中的所有元素。例如，如果 epoch 是 1.23，vals 是 [4.56, 7.89]，那么元组是 (1.23, 4.56, 7.89)。
            # %操作符：这个操作符用于将元组中的值插入到前面创建的 '%g,' * n 字符串中的 %g 占位符中。例如，如果 n 是 3，那么结果字符串可能是 '1.23,4.56,7.89,'。
            # .rstrip(',') 是一个字符串方法，用于从字符串的末尾移除指定的字符（或字符集）。在这里，它移除了末尾的逗号，所以字符串变为 '1.23,4.56,7.89'（没有末尾的逗号）。
            #+ '\n' 在格式化后的字符串末尾添加了一个换行符，这样下一次写入时内容会出现在新的一行。

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi    #保留小的
            #wandb_logger.end_epoch(best_result=best_fitness == fi)
            # fi = MRresult[0]
            # if fi < best_fitness:
            #     best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                #torch.save() 是一个用于保存模型、张量、字典或其他Python对象到磁盘的函数。
                # 这里的 ckpt 很可能是一个包含模型状态字典（state_dict）、优化器状态、训练步数等信息的字典或对象，
                # 而 last 是你想要保存这个文件的路径和文件名。
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    t1 = time.time()
    t = t1 - t0
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(file=save_dir / 'results.csv')  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        for m in (last, best) if best.exists() else (last):  # speed, mAP tests
            #if best.exists():
            #   for m in (last, best):
            #       # ... do something with m ...
            #else:
            #   for m in (last,):
            #       # ... do something with m ...

            results, _, MRresult, _ = test.test(opt.data,
                                                batch_size=1,
                                                imgsz=imgsz_test,
                                                conf_thres=0.001,
                                                iou_thres=0.5,
                                                model=attempt_load(m, device).half(),
                                                single_cls=opt.single_cls,
                                                dataloader=testloader,
                                                save_dir=save_dir,
                                                save_txt=True,
                                                save_conf=True,
                                                save_json=False,
                                                plots=False,
                                                is_coco=is_coco,
                                                labels_list=labels_list,
                                                verbose=nc > 1,
                                                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers，从模型或函数中“剥离”或“移除”与其关联的优化器状态。
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/best20240831_flir_danetv2/weights/best.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_transfusion_FLIR.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR-align-3class.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')  #action='store_true'：这指定了当命令行中出现这个选项时，argparse应该如何处理它。'store_true'意味着如果用户在命令行中指定了这个选项（例如，通过输入--noautoanchor），那么对应的值会被设置为True。如果用户没有指定这个选项，那么默认情况下它的值会是False（或者更具体地说，它不会被包含在解析得到的命名空间中）。
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')  #bucket是一个用于存储数据的命名空间。你可以把它想象成一个文件系统的目录，但是存储桶是全局唯一的，并且与特定的 Google Cloud 项目相关联
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=1, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()

    #opt.rect = False

    # FQY  Flag for visualizing the paired training imgs
    #定义了一个全局字典，存储一些数据
    global_var._init()
    global_var.set_value('flag_visual_training_dataset', False)

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  #分佈式系統設置節點數

    #在分布式训练的场景中，global_rank是一个非常重要的标识符，因为它允许每个进程或节点知道自己在整个训练集群中的位置。这对于确保数据同步、梯度聚合以及任何其他需要跨节点通信的操作都是至关重要的。
    #默认值-1通常表示该进程或节点不是分布式训练的一部分，或者它在尝试确定其排名时遇到了问题。在真实的分布式训练脚本中，如果global_rank被设置为-1，则可能需要额外的逻辑来处理这种情况，例如退出进程或输出错误消息。
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    #在分布式训练中，多个进程（可能位于同一台机器上或跨越多台机器）可能同时运行，并共享训练数据。为了区分这些进程并管理它们的日志输出，通常会为每个进程分配一个唯一的标识符（在这里是 global_rank）。
    set_logging(opt.global_rank)  #用于设置或初始化日志记录（logging）。opt.global_rank 是一个参数，通常用于标识当前进程或设备在全局环境中的排名或索引。
    if opt.global_rank in [-1, 0]: #-1表示非分布式模式，而是单设备或单进程训练。0表示主进程
        # check_git_status()
        check_requirements()
    # W&B 是一个用于机器学习实验跟踪、数据版本控制和模型管理的工具。check_wandb_resume 函数可能会返回一个 wandb_run 对象（如果找到了先前运行的实验）或 None（如果没有找到或不应该恢复）。
    # Resume,Weights & Biases (WandB)，用于检查是否需要从先前的运行（run）中恢复或开始一个新的运行。
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run。opt.resume 表示是否应该尝试恢复先前的训练或实验。 wandb_run 检查是否从 check_wandb_resume 返回的一个有效对象。
        #意味着代码将尝试执行某种恢复操作，但由于某种原因（例如，W&B 没有先前的实验可恢复，或者用户/配置指示应该恢复但没有找到有效的 W&B 实验），它可能需要进行不同的恢复逻辑或简单地开始一个新的实验。
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            #yaml.safe_load(f) 是 PyYAML 库中的一个函数，用于从文件对象 f 中安全地加载 YAML 格式的数据。
            #argparse.Namespace对象通常是由argparse.ArgumentParser的parse_args()方法自动创建的，用于存储从命令行解析得到的参数。
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace。*表示收集参数，**表示收集关键字参数
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate#在某些编程语言或脚本中，特别是在继续长行或表达式时，\ 可以用作行继续符（line continuation character）。
            #opt.cfg = ''
            #opt.weights = ckpt
            #opt.resume = True
            #opt.batch_size = opt.total_batch_size
            #opt.global_rank = apriori[0]  # 这将是1
            #opt.local_rank = apriori[1]   # 这将是2
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)。将创建的新列表（即包含opt.img_size[-1]重复了(2 - len(opt.img_size))次的列表）的元素添加到opt.img_size的末尾
        #如果opt.img_size是[64, 128, 256]（长度大于2），这行代码也不会执行任何操作，因为(2 - len(opt.img_size))将为负数，而负数的乘法在Python中会得到一个空列表。但是，通常这种情况不会是一个问题，因为extend()方法对于空列表的调用是安全的，它不会改变原始列表。

        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    #分布式训练.使用 opt.local_rank 来确定哪个GPU应该被当前进程使用.如果 opt.local_rank 不等于 -1，那么模型将被移动到与当前进程的本地排名相对应的GPU上。如果 opt.local_rank 等于 -1，那么模型将被移动到CPU上
    #进程0（在GPU 0上运行）的 opt.local_rank 为 0
    #进程1（在GPU 1上运行）的 opt.local_rank 为 1
    #进程2（在GPU 2上运行）的 opt.local_rank 为 2
    #进程3（在GPU 3上运行）的 opt.local_rank 为 3

    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size   # // 是一个整数除法运算符。它返回两个数相除后的整数部分，忽略小数部分。

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps。safe_load()只支持基本的数据类型，如列表、字典、字符串、整数、浮点数、布尔值等，并且不允许执行任何Python代码或自定义函数。这是为了防止加载恶意或不受信任的YAML数据时可能导致的安全问题。

    # Train
    logger.info(opt)  #记录日志消息。.info()方法被调用时，opt的内容会被转换为字符串（如果它不是字符串的话），并作为日志消息的一部分被记录
    if not opt.evolve:   #超参数不进化
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

            #try:
            train_rgb_ir(hyp, opt, device, tb_writer)
            #except RuntimeError as exception:
            #   if "out of memory" in str(exception):
            #        print("WARNING:out of memory")
            #        if hasattr(torch.cuda,'empty_cache'):
            #            torch.cuda.empty_cache()
            #    else:
            #        raise exception

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve。_ 是一个常用的占位符变量名，表示在这个循环中并不关心该变量的具体值。range(300) 会生成一个从0到299（包含0但不包含300）的整数序列。这样循环通常用于需要重复执行某个操作或代码块300次的场景，但不需要在每次迭代中引用当前的迭代索引。
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
