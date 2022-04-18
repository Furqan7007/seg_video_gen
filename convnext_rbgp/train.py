import argparse
import copy
import os
import os.path as osp
import time
import json
import math 

import mmcv
import mmcv_custom
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

# from backbone import beit
from backbone import convnext

# import lmodels
import pruners.BlockPruner
import pruners.HbPruner
import pruners.RmbPruner
import pruners.RmcdbPruner
import pruners.SRMBRepMasker
import pruners.GroupingPruner

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--train_config', help='train config file path')
    parser.add_argument('--pr-config-path', type = str, help='pruner config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=2)

    parser.add_argument("--dataset", type=str, default="cityscapes", help="Dataset to use")
    # parser.add_argument("--exp_dir", type=str, default=".", help="Path to experiment directory", dest="exp_dir")
    parser.add_argument("--input_size", type=str, default="512X512")

    # Pruning related
    parser.add_argument("--mc_pruning", action="store_true", help="Enable model compression using pruning")
    # parser.add_argument("--pr-base-model", type=str, help="Path to base dense model", default=None)
    parser.add_argument("--pr_config_path", type=str, help="Path to pruning configuration file", default=None)
    parser.add_argument("--pr_static", action="store_true", help="Randomly generates structure instead of pruning")

    parser.add_argument("--sparsity", type = str, default = '', help="Level of sparsity applied")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.train_config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # input_size = args.input_size
    # print(f"Input size is {input_size}")
    # # batch_size = args.batch_size
    # # num_workers = args.workers

    # inp_ht = int(input_size.split("X")[0])
    # inp_wdth = int(input_size.split("X")[1])
    # sparsity = args.sparsity
    # num_classes = args.classes
    #print(' '.join(sys.argv))
    # if not os.path.isdir(args.exp_dir):
    #     os.mkdir(args.exp_dir)
    
    for k, v in args.__dict__.items():
        print(k, ':', v)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.train_config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.train_config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.train_config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)
    model.to(device)
    # summary(model, (3,inp_ht,inp_wdth))
    # criterion.cuda()    

    pruner = None
    pruner_type = ""
    if args.mc_pruning:
        
        config_path = args.pr_config_path
        print(f"Config file is {config_path}")
        print("Using "+config_path+" configuration file for generating structure.")
        # import json
        with open(args.pr_config_path) as json_file:
            pruner_type = json.load(json_file)["pruner_type"]
        if pruner_type == "block":
            pruner = pruners.BlockPruner.BlockPruner(config_path)
        elif pruner_type == "srmbrep":
            pruner = pruners.SRMBRepMasker.SRMBRepMasker(config_path)
        else:
            print("Unsupported pruner ", pruner_type)
            exit(-1)

        # Pruning related
        pruner.generate_masks(model, is_static=args.pr_static, verbose=True)
        pruner.print_stats()
    
    if args.mc_pruning and args.pr_static:
        with torch.no_grad():
            for layer in pruner.mask_dict:
                print(f"layer name {layer}")
                # Get tensor and mask
                tensor =  model.state_dict()[layer]
                print(f"tensor {tensor.shape}")
                mask   = pruner.mask_dict[layer]
                print(f"mask shape {mask.shape}")

                # Applying mask
                tensor.mul_(mask)

                #Create small tensor
                nnz_count = torch.sum(mask != 0).item()
                print(f"nnz_count {nnz_count}")
                n = nnz_count//mask.shape[1]
                print(f"n value is {n}")
                small_tensor = torch.zeros(nnz_count, dtype=tensor.dtype,
                                            layout=tensor.layout, device=tensor.device)
                
                print(f"small_tensor shape {len(small_tensor.shape)}")

                if len(tensor.shape) == 2:
                    print("Reinitializing FC {} wrt sparsity".format(layer))
                    small_tensor.normal_(0, 0.01)
                else:
                    print("Reinitializing CONV {} wrt sparsity".format(layer))
                    small_tensor.normal_(0, math.sqrt(2. / n))

                # Distribute the values to big tensor
                tensor[torch.nonzero(mask, as_tuple=True)] = small_tensor.flatten()

    if args.mc_pruning:
        # if not args.pr_static:
        #     print("Base line dense accuracy")
        #     validate(val_loader, model, criterion, args)
        print("Applying masking before training begins")
        pruner.apply_masks(model)

    datasets = [build_dataset(cfg.data.train)]
    print(f"Dataset for training {datasets}")
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
