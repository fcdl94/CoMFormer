#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N 15-1_ContM2F
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
#cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 1 CONT.MODE disjoint SEED 42"
#task=voc_15-5-dis
cont_args="CONT.BASE_CLS 15 CONT.INC_CLS 1 CONT.MODE overlap SEED 42"
task=voc_15-5-ov

#name=PerPixel_4L
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
#comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
#inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 1000 "

name=MF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 20"
meth_args="${meth_args} MODEL.MASK_FORMER.DICE_WEIGHT 1. MODEL.MASK_FORMER.CLASS_WEIGHT 1. MODEL.MASK_FORMER.MASK_WEIGHT 0."
comm_args="--dist-url tcp://127.0.0.1:${port} OUTPUT_DIR ${base} ${meth_args} ${cont_args}"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 5000 SOLVER.BASE_LR 0.00005"

python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_FT
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_F_FT MODEL.MASK_FORMER.FOCAL True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE CONT.DIST.UCE True
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_F_UCE MODEL.MASK_FORMER.FOCAL True CONT.DIST.UCE True

for t in 2 3 4 5; do
  inc_args="CONT.TASK ${t} SOLVER.MAX_ITER 1000 SOLVER.BASE_LR 0.00005"
  python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_FT
  python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_F_FT MODEL.MASK_FORMER.FOCAL True
  python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_UCE CONT.DIST.UCE True
  python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_F_UCE MODEL.MASK_FORMER.FOCAL True CONT.DIST.UCE True
done
