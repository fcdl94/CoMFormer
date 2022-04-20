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

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output_inc
#base=output_inc/voc_15-5


name=MF_4L_dice2
comm_args="OUTPUT_DIR ${base} MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.DICE_WEIGHT 0. MODEL.MASK_FORMER.DEC_LAYERS 4" # CONT.MODE disjoint"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${name}/step0/model_final.pth TEST.EVAL_PERIOD 1000 SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} NAME ${name} ${comm_args} CONT.TASK 0

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_FT ${comm_args}
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} NAME ${name}_UCE_LKD100 ${comm_args} ${inc_args} CONT.DIST.UCE False CONT.DIST.KD_WEIGHT 100.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} NAME ${name}_UCE_UKD100 ${comm_args} ${inc_args} CONT.DIST.UCE True CONT.DIST.UKD True CONT.DIST.KD_WEIGHT 100.

python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE_LWFeos01_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True ${comm_args}
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE_UKDeos01_100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.UKD True ${comm_args}

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE_UKD100 CONT.DIST.KD_WEIGHT 100. CONT.DIST.UCE True CONT.DIST.UKD True ${comm_args}
