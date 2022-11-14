#!/bin/bash
#PBS -l select=1:ncpus=4:mem=20GB:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N TestM2F
#PBS -q gpu
module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

port=$(python get_free_port.py)

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output_inc
cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE overlap SEED 42"
task=voc_15-5-ov
#cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE disjoint SEED 42"
#task=voc_15-5-dis

#nq=20
#name=MF_4Ln_121_${nq}q
name=MF_4L
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
#meth_args="${meth_args} MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${nq}"
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 5000 SOLVER.BASE_LR 0.00005"

# TODO
# The main issues are:
# (1) Overconfidence on the prediction (classes are predicted with high confidence even on similar objects, harming the knowledge distillation and pseudo-labeling),
#   (1.a) Introducing a technique to reduce over-confidence (e.g. focal loss, balanced classifier, label smoothing)
#   (1.b) Removing the masked-cross attention - maybe the context is important to identify the right class ??
#   (1.c) Use SGD instead of Adam and/or introducing more regularization (such as MixUp)
# (2) We have multiple masks predicted at the same pixels even with a different class (i.e. we may have cow and horse in the same image with an overlapping mask)
#   (2.a) Increase the weight of the classification loss (matching should be more class-focused, fixing a single query masks for a class)
#         -> Since I noted BCE was too high on VOC, I tried by removing it.
#   (2.b) Use softmax instead of sigmoid in the mask losses -> softmax will force only one mask per every pixel, solving the issue (BUT, requires to change some M2F hyper-parameters)
# To alleviate forgetting we may also think to free the learnable queries (even from the beginning, the performance is close)


#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} CONT.TASK 0 NAME ${name}_cosine CONT.COSINE True

#name=${name}_cosine
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 5000 SOLVER.BASE_LR 0.00005"

python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_WA_FT CONT.WA_STEP 100
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_WA_PSEUDO CONT.WA_STEP 100 CONT.DIST.PSEUDO True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_WA_PSEUDO_KD CONT.WA_STEP 100 CONT.DIST.PSEUDO True CONT.DIST.KD_WEIGHT 1.
