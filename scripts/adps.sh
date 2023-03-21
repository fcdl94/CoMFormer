#!/bin/bash

cfg_file=configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_90k.yaml
base=ade_ps

name=MxF
meth_args="MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"

### OFFLINE ###
cfg_file=configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml
name=MxF_BS8
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True "

python train_net.py --resume --dist-url tcp://127.0.0.1:${port} --num-gpus 1 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${meth_args} #  MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 # INPUT.IMAGE_SIZE 512 # MODEL.MASK_FORMER.FOCAL True

### 100-50 ###
cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 50 CONT.MODE overlap SEED 42"
task=mya-pan_100-50-ov

comm_args="OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 20000 SOLVER.BASE_LR 0.00005"

python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD5Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 5.0 CONT.DIST.UKD True CONT.DIST.KD_REW True
