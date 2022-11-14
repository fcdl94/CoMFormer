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

cfg_file=configs/ade20k/semantic-segmentation/maskformer2_R101_bs16_90k_b.yaml
base=ade_ss
# MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1. MODEL.MASK_FORMER.MASK_WEIGHT 0.0 MODEL.MASK_FORMER.FOCAL True
cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap SEED 42"
task=mya_100-50-ov-b

#name=MxF
#comm_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1. MODEL.MASK_FORMER.MASK_WEIGHT 0.0 MODEL.MASK_FORMER.FOCAL True"

#python train_net.py --dist-url tcp://127.0.0.1:${port} --num-gpus 4 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${comm_args}
#python train_net.py --num-gpus 4 --eval-only --config-file ${cfg_file} OUTPUT_DIR ${base}/${name} ${comm_args} MODEL.WEIGHTS ckpt/coco/model_final_3c8ec9.pkl
#python train_net.py --resume --num-gpus 2 --config-file ${cfg_file}  OUTPUT_DIR ${base}/${name}_ce2_dice0_mask1 ${comm_args} MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1. MODEL.MASK_FORMER.TEST.SEM_AS_PANO True

#name=PerPixel
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
name=MF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False" # MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"
#name=MxF
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"
#meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1.  MODEL.MASK_FORMER.MASK_WEIGHT 0."

#meth_args="${meth_args} MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${nq}"
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 5000 SOLVER.BASE_LR 0.00005"

python train_inc.py --resume --num-gpus 2 --config-file ${cfg_file} ${comm_args} CONT.TASK 0 NAME ${name}
