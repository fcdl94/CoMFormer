#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16GB:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ADEP
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

export WANDB_API_KEY="e8c87155059d91add34a5ace9e134b018e521377"
port=$(python get_free_port.py)

cfg_file=configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_small_bs16_160k.yaml
base=ade_ps

#name=MF
name=Swin_MxF
meth_args="MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"

### OFFLINE ###
#
#cfg_file=configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml
#name=MxF_BS8
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True "

#python train_net.py --resume --dist-url tcp://127.0.0.1:${port} --num-gpus 1 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${meth_args} #  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 # INPUT.IMAGE_SIZE 512 # MODEL.MASK_FORMER.FOCAL True


### 100-50 ###
#cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap SEED 42"
#task=mya-pan_100-50-ov
#
#comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
#inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"
#
#python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD1Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 0.5 CONT.DIST.UKD True CONT.DIST.KD_REW True

#python train_inc.py --resume --num-gpus 4 --config-file ${cfg_file} ${comm_args} CONT.TASK 0 NAME ${name} SOLVER.MAX_ITER 80000

#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD5Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 5.0 CONT.DIST.UKD True CONT.DIST.KD_REW True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD1_Rew CONT.DIST.KD_WEIGHT 1.0 CONT.DIST.KD_REW True  CONT.DIST.UKD True

## 100-10 ###
#cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 10 CONT.MODE overlap SEED 42"
#task=mya-pan_100-50-ov
#
#comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
#inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 4000 SOLVER.BASE_LR 0.00005"
#
##python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD50_Rew CONT.DIST.KD_WEIGHT 50.0 CONT.DIST.KD_REW True CONT.DIST.UKD True # CONT.DIST.UCE True
#python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10.0 CONT.DIST.UKD True CONT.DIST.KD_REW True
#
#for t in 2 3 4 5; do
#  inc_args="CONT.TASK ${t} SOLVER.MAX_ITER 4000 SOLVER.BASE_LR 0.00005"
#python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10.0 CONT.DIST.UKD True CONT.DIST.KD_REW True
#done

## 50-50 ###
#cont_args="CONT.BASE_CLS 50 CONT.INC_CLS 50 CONT.MODE overlap SEED 42"
#task=mya-pan_50-50-ov
#comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
#
#inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_KD05Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 0.5 CONT.DIST.KD_REW True
#inc_args="CONT.TASK 2 SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_KD05Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 0.5 CONT.DIST.KD_REW True

## 100-5 ###
cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap SEED 42"
task=mya-pan_100-50-ov

comm_args="--dist-url tcp://127.0.0.1:${port} WANDB False OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 0.00005"

python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10.0 CONT.DIST.UKD True CONT.DIST.KD_REW True

for t in 2 3 4 5 6 7 8 9 10; do
  inc_args="CONT.TASK ${t} SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 0.00005"
python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10.0 CONT.DIST.UKD True CONT.DIST.KD_REW True

done
