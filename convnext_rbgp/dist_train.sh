#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --job-name=convnext_pruner
#SBATCH --output=log_outputs/convnext%J.out
#SBATCH --mail-user=furqan.shaik@research.iiit.ac.in

source activate open-mmlab

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2 

scratch_dir=/ssd_scratch/cvit/furqan.shaik
mkdir -p ${scratch_dir}

if [ ! -f "${scratch_dir}/cityscapes/cityscapes.tar" ]; then
# Loading data from dataset to scratch
	rsync -a furqan.shaik@ada:/share1/dataset/cityscapes  ${scratch_dir}/
	# rsync -a furqan.shaik@ada:/share3/furqan.shaik/S2021/drn/datasets/cityscapes/* ${scratch_dir}/cityscapes/cityscapes/
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

CONFIG=configs/cityscapes_convnext/upernet_convnext_tiny_512_320k_cityscapes_ms.py
work_dir=work_dirs/upernet_convnext_ti_512x512_320k_cityscapes_ms
pretrained_model=pretrained_models/convnext_tiny_1k_224.pth
GPUS=2
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py --work_dir=$work_dir --train_config $CONFIG --launcher pytorch --seed 0 --deterministic --options model.pretrained=${pretrained_model}

# python -m torch.distributed.launch --nproc_per_node=2 train.py --train_config configs/cityscapes_convnext/upernet_convnext_tiny_512_320k_cityscapes_ms.py --launcher pytorch --seed 0 --deterministic --options model.pretrained=pretrained_models/convnext_tiny_1k_224.pth
# CUDA_VISIBLE_DEVICES=0 python semantic_seg.py /ssd_scratch/cvit/furqan.shaik/ADE20K/ADEChallengeData2016 \
#          --dataset ade20k \
#          --arch convnext \
#          --exp-dir sparse_experiments/block_ade20k_convnext/sparse_ade20k_convnext_srmbrep_512X512_-1x-1_-1x-1_4x4_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive  \
#          --mc-pruning \
#          --pr-base-model experiments/dense_ade20k_convnext/model_best.pth.tar \
#          --pr-config-path sparse_experiments/block_ade20k_convnext/sparse_ade20k_convnext_srmbrep_512X512_-1x-1_-1x-1_4x4_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/config.json  \
#          --pr-static \
#          --lr 0.01 \
#          --epochs 500 \
#          --input_size 512X512 \
#          --batch-size 12 | tee sparse_experiments/block_ade20k_convnext/sparse_ade20k_convnext_srmbrep_512X512_-1x-1_-1x-1_4x4_0.00-RAMANUJAN_50.00-RAMANUJAN_50.00_collapse_repetitive/log.txt