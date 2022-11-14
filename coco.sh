#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16GB:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N COCO
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

port=$(python get_free_port.py)

cfg_file=configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml
#cfg_file=configs/coco/instance-segmentation/contiunal_r50.yaml
base=coco_is
#cont_args="CONT.BASE_CLS 60 CONT.INC_CLS 10 CONT.MODE overlap SEED 42"
#task=coc_60-10-ov

name=MxF_MonlyD
comm_args="MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.DICE_WEIGHT 5. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 5.0  MODEL.MASK_FORMER.FOCAL True"

python train_net.py --resume --dist-url tcp://127.0.0.1:${port} --num-gpus 4 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${comm_args} INPUT.IMAGE_SIZE 512 # MODEL.MASK_FORMER.FOCAL True

#python train_inc.py --resume --num-gpus 2 --config-file ${cfg_file} ${comm_args} CONT.TASK 0 NAME ${name} INPUT.IMAGE_SIZE 512 SOLVER.IMS_PER_BATCH 16
