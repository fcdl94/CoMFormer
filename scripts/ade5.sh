#!/bin/bash

cfg_file=configs/ade20k/semantic-segmentation/maskformer2_R101_bs16_90k.yaml
base=ade_ss

cont_args="CONT.BASE_CLS 100 CONT.INC_CLS 5 CONT.MODE overlap SEED 42"
task=mya_100-50-ov

#name=PerPixel
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
#name=MF
#meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False"
#name=MxF
name=Swin_MxF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.SOFTMASK True MODEL.MASK_FORMER.FOCAL True"

comm_args="OUTPUT_DIR ${base} ${meth_args} ${cont_args} WANDB False"
inc_args="CONT.TASK 1 CONT.WEIGHTS ${base}/${task}/${name}/step0/model_final.pth SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 0.00005"

python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10. CONT.DIST.UKD True CONT.DIST.KD_REW True

for t in 2 3 4 5 6 7 8 9 10; do
  inc_args="CONT.TASK ${t} SOLVER.MAX_ITER 2000 SOLVER.BASE_LR 0.00005"
  python train_inc.py --num-gpus 4 --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${name}_PSEUDO_T2_UKD10Rew CONT.DIST.PSEUDO True CONT.DIST.PSEUDO_TYPE 1 CONT.DIST.KD_WEIGHT 10. CONT.DIST.UKD True CONT.DIST.KD_REW True
done

