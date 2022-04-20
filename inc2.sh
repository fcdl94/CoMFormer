cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
base=output_inc

name=PerPixel
comm_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL True"
inc_args="CONT.TASK 1 CONT.WEIGHTS output_inc/${name}/step0/model_final.pth TEST.EVAL_PERIOD 1000 SOLVER.MAX_ITER 5000"

#python train_inc.py --num-gpus 2 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${comm_args} CONT.TASK 0
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_LR1e-4_warmup  SOLVER.WARMUP_ITERS 500 ${comm_args} SOLVER.BASE_LR 0.0001
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_LR1e-5_warmup  SOLVER.WARMUP_ITERS 500 ${comm_args} SOLVER.BASE_LR 0.00001
python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_LWF_100_warmup CONT.DIST.KD_WEIGHT 100. SOLVER.WARMUP_ITERS 500  ${comm_args} SOLVER.BASE_LR 0.00001
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE CONT.DIST.UCE True ${comm_args}
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE CONT.DIST.UCE True ${comm_args}
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE_LWF CONT.DIST.KD_WEIGHT 1. CONT.DIST.UCE True ${comm_args}
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} NAME ${name}_UCE_UKD CONT.DIST.KD_WEIGHT 1. CONT.DIST.UCE True CONT.DIST.UKD True ${comm_args}


#name=MF_noBg_Dice0
#comm_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.DICE_WEIGHT 0."
#inc_args="CONT.TASK 1 CONT.WEIGHTS output_inc/${name}/step0/model_final.pth TEST.EVAL_PERIOD 1000"
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${comm_args} CONT.TASK 0
##python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} OUTPUT_DIR ${base} NAME ${name}_LR1e-4 ${comm_args} SOLVER.BASE_LR 0.0001
#
#name=MF_Dice0
#comm_args="MODEL.MASK_FORMER.TEST.MASK_BG True MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.DICE_WEIGHT 0."
#inc_args="CONT.TASK 1 CONT.WEIGHTS output_inc/${name}/step0/model_final.pth TEST.EVAL_PERIOD 1000"
#
#python train_inc.py --num-gpus 2 --config-file ${cfg_file} OUTPUT_DIR ${base} NAME ${name} ${comm_args} CONT.TASK 0
##python train_inc.py --num-gpus 2 --config-file ${cfg_file} ${inc_args} OUTPUT_DIR ${base} NAME ${name}_LR1e-4 ${comm_args} SOLVER.BASE_LR 0.0001
