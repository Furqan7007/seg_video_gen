#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --job-name=convnext_pruner
#SBATCH --output=log_outputs/convnext_2048x1024%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

source activate open-mmlab

module load cuda/11.0
module load cudnn/8-cuda-11.0 

scratch_dir=/ssd_scratch/cvit/furqan.shaik
mkdir -p ${scratch_dir}

if [ ! -f  "${scratch_dir}/cityscapes/cityscapes.tar" ]; then
	# Loading data from dataset to scratch
	rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	rsync -a furqan.shaik@ada:/share3/furqan.shaik/S2021/drn/datasets/cityscapes/* ${scratch_dir}/cityscapes/cityscapes/
	cd ${scratch_dir}/cityscapes/
	tar -xvf cityscapes.tar --strip 1 
	# cd ${scratch_dir}/cityscapes/cityscapes/
	# python3 prepare_data.py gtFine/

	# bash create_lists.sh
	# echo ls
	# Copying necessary scripts
	#cp ${launch_dir}/imagenet-scripts/* ${scratch_dir}/imagenet/
fi

cd ~/M2020/mmsegmentation/tools/convert_datasets

python cityscapes.py ${scratch_dir}/cityscapes/cityscapes

cd ~/M2020/ConvNeXt/semantic_segmentation

# CONFIG=configs/cityscapes_convnext/upernet_convnext_tiny_512_160k_cityscapes_ms.py
# sparse_config_dir=sparse_experiments/all_layers_sparse/sparse_cityscapes_convnext_2048x1024_rbgp_0.00_87.5
# mkdir -p ${scratch_dir}/sparse_cityscapes_convnext_2048x1024_rbgp_0.00_87.5
# work_dir=${scratch_dir}/sparse_cityscapes_convnext_2048x1024_rbgp_0.00_87.5
# pretrained_model=pretrained_models/convnext_tiny_1k_224.pth

CONFIG=configs/cityscapes_convnext_1024x768/upernet_convnext_tiny_1024x768_160k_cityscapes_ms.py
sparse_config_dir=sparse_experiments/just_downsample_sparse/sparse_cityscapes_convnext_1024x768_rbgp_0.00_50
mkdir -p ${scratch_dir}/sparse_cityscapes_convnext_1024x768_rbgp_0.00_50
work_dir=${scratch_dir}/sparse_cityscapes_convnext_1024x768_rbgp_0.00_50
pretrained_model=pretrained_models/convnext_tiny_1k_224.pth
# GPUS=2
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py $CONFIG --launcher pytorch --seed 0 --deterministic

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT train.py \
		 --train_config $CONFIG --launcher pytorch \
		 --seed 0 --deterministic \
         --work_dir=${work_dir} --input_size 1024X768 \
         --mc_pruning --pr_static \
		 --pr_config_path=${sparse_config_dir}/config.json | tee ${sparse_config_dir}/log.txt

# rsync -aP ${work_dir} furqan.shaik@ada.iiit.ac.in:/share3/furqan.shaik/convnext_saved_models/

# python -m torch.distributed.launch --nproc_per_node=1 train.py --train_config configs/cityscapes_convnext/upernet_convnext_tiny_512_160k_cityscapes_ms.py --launcher pytorch --seed 0 --local_rank 1 --deterministic --work_dir sparse_experiments/just_downsample_sparse/sparse_cityscapes_convnext_512x512_rbgp_0.00_50.00 --mc_pruning --pr_static --pr_config_path sparse_experiments/just_downsample_sparse/sparse_cityscapes_convnext_512x512_rbgp_0.00_50.00/config.json | tee sparse_experiments/just_downsample_sparse/sparse_cityscapes_convnext_512x512_rbgp_0.00_50.00/log.txt

# python -m torch.distributed.launch --nproc_per_node=2 train.py \
# 		 --train_config configs/cityscapes_convnext/upernet_convnext_tiny_512_160k_cityscapes_ms.py --launcher pytorch \
# 		 --seed 0 --deterministic \
#          --work_dir=${sparse_config_dir}  \
#          --mc_pruning --pr-static --options model.pretrained=${pretrained_model} \
# 		 --pr_config_path=${sparse_config_dir}/config.json | tee ${sparse_config_dir}/log.txt