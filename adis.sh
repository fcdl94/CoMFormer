#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N ADE
#PBS -q gpu

module load anaconda/3.2020.2
module load devtoolset-7/gcc-7.3.1
module load openmpi/4.0.5/gcc7-ib
source activate /home/fcermelli/.conda/envs/m2f/
cd /work/fcermelli/fcdl/Mask2Former

port=$(python get_free_port.py)

cfg_file=configs/ade20k/instance-segmentation/maskformer2_R50_bs16_160k.yaml
base=ade_is

#name=PerPixel
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
#name=MF_200q
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
name=MxF_noFocal
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL False"
#meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 5. MODEL.MASK_FORMER.CLASS_WEIGHT 2.  MODEL.MASK_FORMER.MASK_WEIGHT 0."

python train_net.py --resume --dist-url tcp://127.0.0.1:${port} --num-gpus 4 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${meth_args} # MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 200 # INPUT.IMAGE_SIZE 512 # MODEL.MASK_FORMER.FOCAL True
