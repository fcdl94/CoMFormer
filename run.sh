#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N M2F_IS
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

port=$(python get_free_port.py)

#cfg_file=configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
#base=coco_is

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=voc_full
comm_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.0"
comm_args="${comm_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1.  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 20"
comm_args="--dist-url tcp://127.0.0.1:${port} ${comm_args}"

name=M2F_softmask_A5G2M01
python train_net.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} OUTPUT_DIR ${base}/${name} NAME ${name} MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True MODEL.MASK_FORMER.FOCAL_ALPHA 5. MODEL.MASK_FORMER.MASK_WEIGHT 0.1
name=M2F_softmask_A5G2M03
python train_net.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} OUTPUT_DIR ${base}/${name} NAME ${name} MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True MODEL.MASK_FORMER.FOCAL_ALPHA 5. MODEL.MASK_FORMER.MASK_WEIGHT 0.3
name=M2F_softmask_A5G2M05
python train_net.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} OUTPUT_DIR ${base}/${name} NAME ${name} MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True MODEL.MASK_FORMER.FOCAL_ALPHA 5. MODEL.MASK_FORMER.MASK_WEIGHT 0.5
name=M2F_softmask_A5G2M1
python train_net.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} OUTPUT_DIR ${base}/${name} NAME ${name} MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True MODEL.MASK_FORMER.FOCAL_ALPHA 5. MODEL.MASK_FORMER.MASK_WEIGHT 1.0


 # MODEL.MASK_FORMER.NO_OBJECT_WEIGHT 0.2 # SOLVER.BASE_LR 0.0005
#python train_net.py --num-gpus 4 --eval-only --config-file ${cfg_file} OUTPUT_DIR ${base}/${name} ${comm_args} MODEL.WEIGHTS ckpt/coco/model_final_3c8ec9.pkl
#python train_net.py --resume --num-gpus 2 --config-file ${cfg_file}  OUTPUT_DIR ${base}/${name}_ce2_dice0_mask1 ${comm_args} MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1. MODEL.MASK_FORMER.TEST.SEM_AS_PANO True