import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os

from detectron2.modeling import build_model
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts, DatasetMapper
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from continual.config import add_continual_config

setting = 'voc_15-5-dis'
config = "configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml"
name = "MF_4L"
step = 0
print(f"================ Computing model of {name}!  ================")

def get_data(image_id):
    path = f"datasets/PascalVOC2012/JPEGImages/{image_id}.jpg"
    image = utils.read_image(path, format=cfg.INPUT.FORMAT)
    return image


def to_tensor(image):
    return torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))


def predict(model, img):
    with torch.no_grad():
        out = model(img)

        logits, mask = out['outputs']['pred_logits'][0], out['outputs']['pred_masks'][0]

        img_size = img[0]['image'].shape[-2], img[0]['image'].shape[-1]
        mask_pred_results = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=img_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pred = model.semantic_inference(logits, mask_pred_results).argmax(dim=0)

    ql, qm = out['outputs']['aux_outputs'][0]['pred_logits'][0], out['outputs']['aux_outputs'][0]['pred_masks'][0]
    return pred, logits, mask, (ql, qm)


print('Collecting IDs')
split = 'datasets/PascalVOC2012/splits/val.txt'
with open(os.path.join(split), "r") as f:
    image_ids = [x[:-1].split(' ')[1].split('/')[-1][:-4] for x in f.readlines()]
print(f"Found {len(image_ids)} validation images.")


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_continual_config(cfg)

cfg.merge_from_file(config)

if cfg.CONT.MODE == 'overlap':
    cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-ov"
elif cfg.CONT.MODE == "disjoint":
    cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-dis"
else:
    cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-seq"

cfg.OUTPUT_ROOT = cfg.OUTPUT_DIR
cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + cfg.TASK_NAME + "/" + cfg.NAME + "/step" + str(cfg.CONT.TASK)

mapper = DatasetMapper(cfg, False)

dataset = get_detection_dataset_dicts(
    cfg.DATASETS.TEST[0],
    filter_empty=False,
    proposal_files=None,
)

cfg.CONT.TASK = step
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.MASK_FORMER.TEST.MASK_BG = False
cfg.MODEL.MASK_FORMER.PER_PIXEL = False
cfg.MODEL.RESNETS.NORM = "BN"

model = build_model(cfg)
print("Loading model weights")
model_weights = torch.load(f"output_inc/{setting}/{name}/step{step}/model_final.pth", map_location='cpu')
model.load_state_dict(model_weights['model'])
print("Model ready!")

model.model_old = True
model = model.eval()
model = model.to("cuda")

res = {}
for image_id in tqdm(image_ids):
    pil_img = get_data(image_id)
    img = [{"image": to_tensor(pil_img)}]
    pred, logits, masks, q = predict(model, img)
    entry = {"score": logits.cpu().numpy(), "masks": masks.cpu().numpy()}
    res[image_id] = entry

file_name = f"output_inc/{setting}/{name}/step{step}/full_prediction.pkl"
with open(file_name, "wb") as f:
    pkl.dump(res, f)
