import torch
import json
import pycocotools.mask as mask_util
from PIL import Image
import mask2former
from detectron2.data import MetadataCatalog
import numpy as np
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import pickle as pkl

from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetMapper
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former import add_maskformer2_config
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

from continual.config import add_continual_config
from continual.data import ContinualDetectron, class_mapper

setting = 'voc_15-5-ov'

# res
print('Collecting IDs')

split = 'datasets/PascalVOC2012/splits/val.txt'
# split = 'datasets/PascalVOC2012/splits/train_aug.txt'
with open(os.path.join(split), "r") as f:
    image_ids = [x[:-1].split(' ')[1].split('/')[-1][:-4] for x in f.readlines()]
print(f"Found {len(image_ids)} validation images.")

config = "configs/voc/semantic-segmentation/maskformer2_R101_bs16_20k.yaml"

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


def get_data(image_id):
    path = f"datasets/PascalVOC2012/JPEGImages/{image_id}.jpg"
    image = utils.read_image(path, format=cfg.INPUT.FORMAT)
    return image


def get_lbl(image_id):
    lbl = np.array(Image.open(f"datasets/PascalVOC2012/SegmentationClassAug/{image_id}.png"))
    return lbl


def to_tensor(image):
    return torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))


def predict(model, img, as_old=False):
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

        if as_old:
            logits[:, 15:-1] = 0.
        pred = model.semantic_inference(logits, mask_pred_results).argmax(dim=0)

    ql, qm = out['outputs']['aux_outputs'][0]['pred_logits'][0], out['outputs']['aux_outputs'][0]['pred_masks'][0]
    return pred.to("cpu"), logits, mask, (ql, qm)


def predict_oracle(model, img, lbl):
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

        cl = np.unique(lbl) - 1
        #         print(cl)
        predicted_classes = logits.argmax(dim=1)
        #         print(predicted_classes)
        keep = torch.tensor([pc.item() in cl for pc in predicted_classes]).float().to(mask_pred_results.device)
        mask_pred_results += (1 - keep.view(-1, 1, 1)) * -9000000
        #         print(keep)
        pred = model.semantic_inference(logits, mask_pred_results).argmax(dim=0)

    ql, qm = out['outputs']['aux_outputs'][0]['pred_logits'][0], out['outputs']['aux_outputs'][0]['pred_masks'][0]
    return pred.to("cpu"), logits, mask, (ql, qm)


def load_model(name, step=1, setting='voc_15-1-ov'):
    cfg.CONT.TASK = step
    cfg.MODEL.DEVICE = "cuda:0"
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
    return model


class Evaluator():
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
            self,
            cfg,
    ):
        self._cpu_device = torch.device("cpu")

        # get some info from config
        self._num_classes = 1 + cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
        self.base_classes = 1 + cfg.CONT.BASE_CLS
        self.novel_classes = cfg.CONT.INC_CLS * cfg.CONT.TASK

        self.old_classes = self.base_classes + (cfg.CONT.TASK - 1) * cfg.CONT.INC_CLS \
            if cfg.CONT.TASK > 0 else 1 + cfg.CONT.BASE_CLS
        self.new_classes = cfg.CONT.INC_CLS if cfg.CONT.TASK > 0 else self.base_classes
        # Background is always present in evaluation, so add +1
        self._order = cfg.CONT.ORDER if cfg.CONT.ORDER is not None else list(range(1, self._num_classes))
        self._name = cfg.NAME
        self._task = cfg.CONT.TASK

        # assume class names has background and it's the first
        self._ignore_label = 255

        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output.to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            # with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
            #     gt = np.array(Image.open(f), dtype=np.int)
            gt = np.array(input, dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

    def evaluate(self):
        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        # prec[acc_valid] = tp[acc_valid] / (pos_pred[acc_valid] + 1e-5)
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        miou_base = np.sum(iou[1:self.base_classes]) / (self.base_classes - 1)
        miou_old = np.sum(iou[1:self.old_classes]) / (self.old_classes - 1)
        miou_new = np.sum(iou[self.old_classes:]) / self.new_classes
        miou_novel = np.sum(iou[self.base_classes:]) / self.novel_classes if self.novel_classes > 0 else 0.

        fg_iou = (np.sum(self._conf_matrix[1:-1, 1:-1]) + self._conf_matrix[0, 0]) / np.sum(self._conf_matrix[:-1, :-1])

        res = {}
        cls_iou = []
        cls_acc = []

        res["mIoU"] = 100 * miou
        res["mIoU_new"] = 100 * miou_new
        res["mIoU_novel"] = 100 * miou_novel
        res["mIoU_old"] = 100 * miou_old
        res["mIoU_base"] = 100 * miou_base

        res["fwIoU"] = 100 * fiou
        res["fgIoU"] = 100 * fg_iou
        for i, name in enumerate(range(self._num_classes)):
            res["IoU-{}".format(name)] = 100 * iou[i]
            cls_iou.append(100 * iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(range(self._num_classes)):
            res["ACC-{}".format(name)] = 100 * acc[i]
            cls_acc.append(100 * acc[i])

        results = {"sem_seg": res}
        return results


if __name__ == "__main__":

    mapper = DatasetMapper(cfg, False)
    dataset = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN[0],
        filter_empty=False,
        proposal_files=None
    )

    cfg.CONT.TASK = 5
    cfg.CONT.BASE_CLS = 15
    cfg.CONT.INC_CLS = 1
    cfg.CONT.MODE = "overlap"
    step = cfg.CONT.TASK

    model = load_model("MF_4Ln_PSEUDO_CE", step=step)
    model.model_old = True

    log_acc = torch.zeros((100, 21))

    evaluator1 = Evaluator(cfg)
    evaluator2 = Evaluator(cfg)

    for image_id in tqdm(image_ids):
        # image_id = image_ids[idx]
        pil_img = get_data(image_id)

        img = [{"image": to_tensor(pil_img)}]
        lbl = get_lbl(image_id)

        pred, logits, _, q = predict_oracle(model, img, lbl)
        pred2, logits2, _, q2 = predict(model, img)

        log_acc += logits2.softmax(dim=1).cpu()

        evaluator1.process(lbl, pred)
        evaluator2.process(lbl, pred2)

    print(evaluator1.evaluate())
    print(evaluator2.evaluate())

    log_acc = log_acc[:, :20]
    log_acc = log_acc / log_acc.sum(dim=0)
    for i in range(20):
        print(f"{i}: ", end="")
        for j in range(100):
            if log_acc[j, i] > 0.05:
                print(j, round(log_acc[j, i].item(), 2), end=", ")
        print()


