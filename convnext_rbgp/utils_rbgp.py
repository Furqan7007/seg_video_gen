import torch
import sys
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import os
import numpy as np
import math
# Loading torch models and local models
import torchvision.models as models
sys.path.insert(0, '../lmodels')
from lmodels import mmcv_convnext
from lmodels import drn
import lmodels

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def create_model(dataset, arch, pretrained=False):

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    lmodel_names = sorted(name for name in lmodels.__dict__
        if name.islower() and not name.startswith("__")
        and callable(lmodels.__dict__[name]))

    # Fixing number of classes
    if dataset == "imagenet":
        num_classes = 1000
    elif dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "cityscapes":
        num_classes = 19
    elif dataset == "ade20k":
        num_classes = 150
    else:
        print("Invalid dataset")
        exit(-1)

    # create model
    if arch in model_names:
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](num_classes = num_classes, pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch](num_classes = num_classes)
    elif arch in lmodel_names:
        model = lmodels.__dict__[arch](num_classes=num_classes)
    elif arch in ["drn_d_22", "drn_d_38"]:
        model = drn.__dict__.get(arch)(pretrained=pretrained, num_classes=num_classes)
        # pmodel = nn.DataParallel(model)
        # if pretrained_model is not None:
        #     pmodel.load_state_dict(pretrained_model, strict=False)
        base = nn.Sequential(*list(model.children())[:-2])

        seg = nn.Conv2d(model.out_dim, num_classes,
                             kernel_size=1, bias=True)
        softmax = nn.LogSoftmax()
        m = seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        # if use_torch_up:
        #     up = nn.UpsamplingBilinear2d(scale_factor=8)
        # else:
        up = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=num_classes,
                                    bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        #    up = up
    else:
        print("Invalid model name ", arch)
        exit(-1)

    return model

def get_model_information(dataset, arch):
    import collections
    flop_dict = collections.OrderedDict()
    param_dict = collections.OrderedDict()
    parent_lists = collections.OrderedDict()
    child_lists = collections.OrderedDict()

    if "cifar" in dataset:
        if arch == "cifar_resnet18":
            json_fp = "cifar_resnet18.json"
        if arch == "cifar_vgg16_bn":
            json_fp = "cifar_vgg16_bn.json"


    # Reading from json
    import json
    with open(json_fp) as json_file:
        data = json.load(json_file)
        for layer in data:
            linfo = data[layer]
            filter_size = (linfo["ifm"] * linfo["ks"][0] * linfo["ks"][1])
            lflops = (linfo["ofm"] * linfo["oh"] * linfo["ow"]) * filter_size
            flop_dict[layer] = lflops

            lparams = linfo["ofm"] * linfo["ifm"] * linfo["ks"][0] * linfo["ks"][1]
            param_dict[layer] = lparams

            parent_lists[layer] = linfo["parents"]
            child_lists[layer]  = linfo["children"]


    return param_dict, flop_dict, parent_lists, child_lists

