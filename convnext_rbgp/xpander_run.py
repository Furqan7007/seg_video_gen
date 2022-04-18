import os
import itertools
import json
import argparse
import subprocess

import numpy as np
import math

import utils_rbgp
import torch

from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__

from mmseg.models import build_segmentor

def dump_config_file(dataset, cfg, arch, pruner_type, oblock_size, cblock_size, iblock_size, osp, opat, isp, ipat, 
                is_repetitive, collapse_tensor, cross_prob, is_symmetric, pconfig_path):
    
    # Getting the model
    # model = utils_rbgp.create_model(dataset, arch)
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


    rbgp_layers = []
    if arch == "mobilenet_v2":
        # Do not prune, first and last layer
        non_rbgp_layers = ["features.0.0","classifier.1"]
    elif arch in ["drn_d_22", "drn_d_54"]:
        non_rbgp_layers = ["layer0.0","fc"]
    elif arch in ["convnext"]:
        non_rbgp_layers = ["backbone.downsample_layers.0.0","backbone.downsample_layers.0.0.Conv2d","auxiliary_head.conv_seg.Conv2d"]

    # Check validity of non_rbgp_layers
    lnames = []
    for name,module in model.named_modules():
        lnames.append(name)
    assert([_ in lnames for _ in non_rbgp_layers])

     # Selecting layers to do rbgp
    rbgp_layers = []
    for name,module in model.named_modules():
        if type(module) == torch.nn.Linear or (type(module) == torch.nn.Conv2d and module.groups == 1 or "downsample_layers" in name):
            if name not in non_rbgp_layers:
                rbgp_layers.append("layer."+name.lstrip("layer")+".weight")
        elif type(module) in [torch.nn.Linear, torch.nn.Conv2d]:
            non_rbgp_layers.append(name)

    verbose = False
    if verbose:
        print("RBGP layers", rbgp_layers)
        print("Non RBGP layers", non_rbgp_layers)

    import collections
    ls_config = collections.OrderedDict()

    ls_config["obh"] = oblock_size[0]
    ls_config["obw"] = oblock_size[1]
    ls_config["cbh"] = cblock_size[0]
    ls_config["cbw"] = cblock_size[1]
    ls_config["ibh"] = iblock_size[0]
    ls_config["ibw"] = iblock_size[1]
    ls_config["osp"] = osp
    ls_config["opat"] = opat
    ls_config["isp"] = isp
    ls_config["ipat"] = ipat
    ls_config["is_repetitive"] = is_repetitive
    ls_config["collapse_tensor"] = collapse_tensor

    ls_config["cross_prob"] = cross_prob
    ls_config["is_symmetric"] = is_symmetric

    ls_config["layer_set"] = rbgp_layers

    # Constructing json
    data = collections.OrderedDict()
    data["pruner_type"] = "srmbrep"
    data["configs"] = [ls_config]

    ####### Specially handling 4x4 for sparsites > 87.5 #######

    if  isp >= 0.875 and \
        iblock_size[0] == 4 and iblock_size[1] == 4:

        import copy
        if arch=="drn_d_22":

        # Clone ls_config
        
            if isp==0.875:
                rbgp_layers_2x2 = ["layer.1.0.weight",
                                    "layer.2.0.weight",
                                    "layer.3.0.downsample.0.weight"]
            elif isp==0.9375:
                rbgp_layers_2x2 = ["layer.1.0.weight",
                                    "layer.2.0.weight",
                                    "layer.3.0.conv1.weight",
                                    "layer.3.0.downsample.0.weight",
                                    "layer.5.0.downsample.0.weight"]
        
        elif arch =="drn_d_54":
            if isp==0.875:
                rbgp_layers_2x2 = ["layer.1.0.weight",
                                    "layer.2.0.weight",
                                    "layer.3.0.downsample.0.weight"]
            elif isp==0.9375:
                rbgp_layers_2x2 = ["layer.1.0.weight",
                                    "layer.2.0.weight",
                                    "layer.3.0.conv1.weight",
                                    "layer.3.0.downsample.0.weight",
                                    "layer.5.0.downsample.0.weight"]


        # Preparing 4x4
        ls_config_4x4 = copy.deepcopy(ls_config)
        rbgp_layers_4x4 = []
        for layer in ls_config["layer_set"]:
            if not layer in rbgp_layers_2x2:
                rbgp_layers_4x4.append(layer)
        ls_config_4x4["layer_set"] = rbgp_layers_4x4

        # Preparing 2x2
        ls_config_2x2 = copy.deepcopy(ls_config)
        ls_config_2x2["layer_set"] = rbgp_layers_2x2
        if isp == 0.875:
            ls_config_2x2["ibh"] = 2
            ls_config_2x2["ibw"] = 2
        else:
            ls_config_2x2["ibh"] = 1
            ls_config_2x2["ibw"] = 1


        data["configs"] = [ls_config_4x4, ls_config_2x2]
        print(data)


    #########################################################

    
    # Writing configuration file
    fh = open(pconfig_path,"w+")
    json.dump(data, fh, indent=4)
    fh.close()


def extract_accuracy(exp_dir):
    import os
    # import torch
    return torch.load(os.path.join(exp_dir,"model_best.pth.tar"))["best_acc1"].item()


from tools.calculate_spectral_gap import extract_spectral_gap
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input_size", type=str, default ='512X512')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # Constants and information
    dataset_path_map = {"imagenet":"~/datasets/imagenet" ,
                        "cifar10":"./data",
                        "cifar100":"./data",
                        "cityscapes":"/ssd_scratch/cvit/furqan.shaik/cityscapes/cityscapes",
                        "ade20k":"/ssd_scratch/cvit/furqan.shaik/ADE20K/ADEChallengeData2016"}
    
    # RBGP fixed configuration
    input_size = args.input_size
    pruner_type = "srmbrep"
    opat = "RAMANUJAN"
    ipat = "RAMANUJAN"
    collapse_tensor = True
    is_repetitive = True
    cross_prob = 0.5
    is_symmetric = False

    # Type of experiments (Training, Finetuning) and arguments
    is_pruning = True 
    is_static = True
    #is_kd = True
    kd_temperature = 4

    # RBGP variable configuration
    # osp_isp_choices = [[0,0.5],[0,0.75],[0,0.875],[0,0.9375]]
    osp_isp_choices = [[0,0.5]]
    block_config_choices = []
    # block_config_choices.append(None)
    # block_config_choices.append([[-1,-1], [-1,-1], [1,1]])
    block_config_choices.append([[-1,-1], [-1,-1], [4,4]])
    dataset_choices = ["cityscapes"]
    #arch = "cifar_wrn_40_4"
    arch = "convnext"
    kd_choices = [False] 

    lr = 0.01
    batch_size = 12
    epochs = 500

    import itertools
    exp_id = 0
    BASE_GPU_ID = 0
    NUM_GPUS = 4
    cur_sbps = []
    for exp_id,exp_config in enumerate(itertools.product(dataset_choices, 
                                        block_config_choices,
                                        osp_isp_choices,
                                        kd_choices)):

        # Decoding the configuration
        dataset, block_config, oisp, is_kd = exp_config
        osp,isp = oisp
        dataset_dir = dataset_path_map[dataset]

        exp_dump_dir = "sparse_experiments/block_{}_{}".format(dataset, arch)
        base_model_path = "experiments/dense_{}_{}/model_best.pth.tar".format(dataset, arch)
        
        if not os.path.exists(exp_dump_dir):
            os.makedirs(exp_dump_dir)

        # Are we providing RBGP configuration from outside ?
        is_rbgp_outside = block_config is None
    
        # Experimernt information
        exp_info = pruner_type
        if is_rbgp_outside:
            #exp_info += "_rbgp"
            exp_info += "_rbgpcum"
        else:
            oblock_size, cblock_size, iblock_size = block_config
            exp_info += "_" + "{}".format(input_size)
            exp_info += "_" + "{}x{}".format(oblock_size[0], oblock_size[1])
            exp_info += "_" + "{}x{}".format(cblock_size[0], cblock_size[1])
            exp_info += "_" + "{}x{}".format(iblock_size[0], iblock_size[1])
        exp_info += "_" + "{:.2f}-{}".format(osp*100, opat)
        exp_info += "_" + "{:.2f}-{}".format(isp*100, ipat)
        
        if cross_prob is not None:
            ### Ramanujan related ####           
            assert(opat == "RAMANUJAN" and ipat == "RAMANUJAN")
            exp_info += "_{:.2f}".format(cross_prob*100)
            if is_symmetric:
                exp_info += "_" + "symmetric"
            ##########################

        if collapse_tensor:
            exp_info += "_" + "collapse"
        if is_repetitive:
            exp_info += "_" + "repetitive"
        if is_kd:
            exp_info += "_kd"
        
        # Name of the experiment
        ename =  "sparse_{}_{}_{}".format(dataset, arch, exp_info)
        exp_dir = os.path.join(exp_dump_dir, ename)
        gpu_id = BASE_GPU_ID + exp_id%NUM_GPUS
        """
        print("{:7.3f}".format(extract_spectral_gap(exp_dir)), end=",")
        print("{:7.3f}".format(extract_accuracy(exp_dir)), end=",")
        if cross_prob == 0.5:
            print()
        continue
        """
        if args.dry_run:
            print("{} {}".format(gpu_id,ename))
            if os.path.exists(exp_dir):
                if os.path.exists(os.path.join(exp_dir, "checkpoint.pth.tar")):
                    print("Remove experiment directory {}".format(exp_dir))
                    print("rm -rf {}".format(exp_dir))
                    print("{:7.3f}".format(extract_accuracy(exp_dir)))
                else:
                    print("Cleaning up empty directory")
                    import shutil
                    shutil.rmtree(exp_dir)
            continue
        
        # Create experiment directory if does not exists.
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        else:
            if os.path.exists(os.path.join(exp_dir, "checkpoint.pth.tar")):
                print("Remove experiment directory {}".format(exp_dir))
                print("rm -rf {}".format(exp_dir))
                exit(-1)

        # Dumping config file into experiment directory
        pconfig_path = os.path.join(exp_dir, "config.json")
        if is_rbgp_outside:
            #cpath = "./rbgp_configs/{}_{:.2f}_{:.2f}.json".format(arch, osp*100, isp*100)
            cpath = "./rbgp_configs/{}_{:.2f}.json".format(arch, isp*100)
            import os
            os.system("cp {} {}".format(cpath, pconfig_path))
        else:
            dump_config_file(dataset, cfg, arch, pruner_type, oblock_size, cblock_size, iblock_size, osp, opat, 
                isp, ipat, is_repetitive, collapse_tensor, cross_prob, is_symmetric, pconfig_path)

        # Model compression args
        mc_args = ""
        # Pruning related
        if is_pruning:
            mc_args += " --mc-pruning --pr-config-path {}".format(base_model_path, pconfig_path)
            if is_static:
                mc_args += "  --pr-static"

        # Knowledge distillatin related
        if is_kd:
            mc_args += " --mc-kd --kd-teacher {} --kd-temperature {}".format(base_model_path, kd_temperature)
        
        # Final command
        cmd = "python semantic_seg.py {} --dataset {} --arch {} --exp-dir {} {} --lr {} --epochs {} --input_size {} --batch-size {} | tee {}/log.txt".\
                    format(dataset_dir, dataset, arch, exp_dir, mc_args, lr, epochs, input_size, batch_size, exp_dir)

        # Executing command
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(gpu_id)+cmd
        pretty_cmd = cmd.replace(" --"," \\\n\t --")+"\n"
        print(pretty_cmd)
        continue
        
        p = subprocess.Popen(cmd, shell=True)
        cur_sbps.append(p)

        if exp_id%NUM_GPUS == NUM_GPUS-1:
            exit_codes = [p.wait() for p in cur_sbps]
            cur_sbps = [] # Emptying the process list