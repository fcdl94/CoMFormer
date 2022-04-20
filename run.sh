#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N M2F
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output
name=M2F
comm_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"

#python train_net.py --num-gpus 2 --resume --config-file ${cfg_file} OUTPUT_DIR ${base}/${name} ${comm_args}
python train_net.py --resume --num-gpus 2 --config-file ${cfg_file}  OUTPUT_DIR ${base}/${name}_ce2_dice0_mask1 ${comm_args} MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1. MODEL.MASK_FORMER.TEST.SEM_AS_PANO True