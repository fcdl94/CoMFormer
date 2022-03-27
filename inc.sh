#python train_net.py --num-gpus 2 --config-file configs/voc/semantic-segmentation/maskformer2_swin_small_bs16_20k.yaml OUTPUT_DIR output/voc_SWIN_S
#python train_inc.py --num-gpus 2 --config-file configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml OUTPUT_DIR output_inc/voc_NODICE_NoMASKBG MODEL.MASK_FORMER.DICE_WEIGHT 0.

cfg_file=configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml
comm_args="MODEL.MASK_FORMER.DICE_WEIGHT 0. CONT.TASK 1 CONT.WEIGHTS output_inc/voc_NODICE_NoMASKBG/model_final.pth TEST.EVAL_PERIOD 100"
base=output_inc/voc_NODICE_NoMASKBG

python train_inc.py --num-gpus 2 --config-file ${cfg_file} OUTPUT_DIR ${base}_LR1e-4/step1 ${comm_args} SOLVER.BASE_LR 0.0001
python train_inc.py --num-gpus 2 --config-file ${cfg_file} OUTPUT_DIR ${base}_LR1e-5/step1 ${comm_args} SOLVER.BASE_LR 0.00001