#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ContM2F
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
#cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE disjoint"
#task=voc_15-5-dis

name=PerPixel_AvgLoss
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"
#
#python train_inc.py --num-gpus 2 --resume --config-file ${cfg_file} ${comm_args} NAME ${name} CONT.TASK 0
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWFwNew_10 CONT.DIST.KD_WEIGHT 10. CONT.DIST.USE_NEW True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWF_1 CONT.DIST.KD_WEIGHT 1.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWF_2 CONT.DIST.KD_WEIGHT 2.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_LWF_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD_10_fix CONT.DIST.KD_WEIGHT 10. CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_10 CONT.DIST.KD_WEIGHT 10. CONT.DIST.UKD True CONT.DIST.UCE True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_20 CONT.DIST.KD_WEIGHT 20. CONT.DIST.UKD True CONT.DIST.UCE True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_10_alpha05 CONT.DIST.KD_WEIGHT 10. CONT.DIST.UKD True CONT.DIST.UCE True CONT.DIST.ALPHA 0.5
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_25_alpha01 CONT.DIST.KD_WEIGHT 25. CONT.DIST.UKD True CONT.DIST.UCE True CONT.DIST.ALPHA 0.1
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_25_alpha05 CONT.DIST.KD_WEIGHT 25. CONT.DIST.UKD True CONT.DIST.UCE True CONT.DIST.ALPHA 0.5
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_10_fix CONT.DIST.KD_WEIGHT 10. CONT.DIST.UCE True CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_25 CONT.DIST.KD_WEIGHT 25. CONT.DIST.UCE True CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_50 CONT.DIST.KD_WEIGHT 50. CONT.DIST.UCE True CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_75 CONT.DIST.KD_WEIGHT 75. CONT.DIST.UCE True CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.UKD True


name=MF_4L
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
#meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
comm_args="--dist-url tcp://127.0.0.1:${port}  OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"

#python train_inc.py --num-gpus 2 --resume --config-file ${cfg_file} ${comm_args} NAME ${name} CONT.TASK 0
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_FT
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE CONT.DIST.UCE True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_10 CONT.DIST.KD_WEIGHT 10. CONT.DIST.UCE True CONT.DIST.UKD True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWF_noDeep_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.KD_DEEP False
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_LWF_noDeep_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.KD_DEEP False
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE_UKD_noDeep_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.UKD True CONT.DIST.KD_DEEP False