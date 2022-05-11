#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ContM2F
#PBS -q gpu
# :scratch=20gb

# setup env
module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

#cp -rL datasets $PBS_SCRATCHDIR
#echo $DETECTRON2_DATASETS
port=$(python get_free_port.py)

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output_inc
cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE overlap SEED 42"
task=voc_15-5-ov
#cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 5 CONT.MODE disjoint SEED 42"
#task=voc_15-5-dis

name=MF_4L
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 2. MODEL.MASK_FORMER.MASK_WEIGHT 1."
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} NAME ${name}_IQ CONT.TASK 0 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5

python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_SCALE2_42 CONT.DIST.PSEUDO True SEED 42 CONT.AUG True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_SCALE2_10 CONT.DIST.PSEUDO True SEED 10 CONT.AUG True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_SSD CONT.DIST.PSEUDO True SEED 42 INPUT.COLOR_AUG_SSD False
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD5_eos1_2 CONT.DIST.UKD True CONT.DIST.KD_WEIGHT 5. CONT.DIST.EOS_POW 1.  SEED 20
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD5_eos1_3 CONT.DIST.UKD True CONT.DIST.KD_WEIGHT 5. CONT.DIST.EOS_POW 1.  SEED 30
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UKD5_eos1_4 CONT.DIST.UKD True CONT.DIST.KD_WEIGHT 5. CONT.DIST.EOS_POW 1.  SEED 40

name=MF_4L_IQ
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000 "
###
##python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_S2 CONT.DIST.PSEUDO True CONT.DIST.SANITY 2. CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5
##python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_S5 CONT.DIST.PSEUDO True CONT.DIST.SANITY 5. CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_KD1 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5 CONT.DIST.KD_WEIGHT 1.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_KD10 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5 CONT.DIST.KD_WEIGHT 10.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_KD75 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5 CONT.DIST.KD_WEIGHT 75.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_KD100 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5 CONT.DIST.KD_WEIGHT 100.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_KD200 CONT.INC_QUERY True MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 5 CONT.DIST.KD_WEIGHT 200.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_S05 CONT.DIST.PSEUDO True CONT.DIST.SANITY 0.5
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_S075 CONT.DIST.PSEUDO True CONT.DIST.SANITY 0.75
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_CE_WM0 CONT.DIST.PSEUDO True CONT.DIST.WEIGHT_MASK 0.
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_S1_A01 CONT.DIST.PSEUDO True CONT.DIST.ALPHA 0.1
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_S1_A05 CONT.DIST.PSEUDO True CONT.DIST.ALPHA 0.5

# CONT.DIST.KD_WEIGHT 5. CONT.DIST.EOS_POW 1. CONT.DIST.UCE True CONT.DIST.UKD True CONT.DIST.USE_NEW True


#name=PerPixel_4L
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
#comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
#inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00001 SOLVER.MAX_ITER 5000"
##
##python train_inc.py --num-gpus 2 --resume --config-file ${cfg_file} ${comm_args} NAME ${name} CONT.TASK 0
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWFwNew_10 CONT.DIST.KD_WEIGHT 10. CONT.DIST.USE_NEW True
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_LWF_1 CONT.DIST.KD_WEIGHT 1.
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
