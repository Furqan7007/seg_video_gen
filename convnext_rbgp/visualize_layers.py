import argparse
import torch
import mmcv
import mmcv_custom
# import torch
# from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
# from mmseg.apis import set_random_seed
# from mmcv_custom import train_segmentor
# from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
# from mmseg.utils import collect_env, get_root_logger
from backbone import convnext

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)

model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))

print(model)

num_layers = 0
print("Inside printing of layers names")
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        print(name, layer)
        num_layers +=1

print(num_layers)

non_rbgp_layers = ["backbone.downsample_layers.0.0.Conv2d","auxiliary_head.conv_seg.Conv2d"]
rbgp_layers = []
for name,module in model.named_modules():
    if type(module) == torch.nn.Linear or (type(module) == torch.nn.Conv2d and module.groups == 1 or "backbone" in name):
        if name not in non_rbgp_layers:
            rbgp_layers.append(name.lstrip("layer")+".weight")
    elif type(module) in [torch.nn.Linear, torch.nn.Conv2d]:
        non_rbgp_layers.append(name)

num_rbgp_layers = 0

for layer in rbgp_layers:
    print(layer)
    num_rbgp_layers +=1
print("Num of RBGP layers {}".format(num_rbgp_layers))

num_non_layers = 0

for layer in non_rbgp_layers:
    print(layer)
    num_non_layers +=1
print("Non rbgp layers {}".format(num_non_layers))