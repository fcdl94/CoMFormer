#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N Cont_M2F
#PBS -q gpu

# setup env
module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former
port=$(python get_free_port.py)

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output_inc
cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE overlap"
task=voc_15-5-ov

name=MF_4L
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
#meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
comm_args="--dist-url tcp://127.0.0.1:${port}  OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"

#python train_inc.py --num-gpus 2 --resume --config-file ${cfg_file} ${comm_args} NAME ${name} CONT.TASK 0

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_FT
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWF_100 CONT.DIST.KD_WEIGHT 100.
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.UKD True