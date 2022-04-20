from .base import BaseDistillation
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.matcher import HungarianMatcher
from .set_criterion import KDSetCriterion


class MaskFormerDistillation(BaseDistillation):
    def __init__(self, cfg, model, model_old):
        super().__init__(cfg, model, model_old)
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        self.use_kd = cfg.CONT.TASK and cfg.CONT.DIST.KD_WEIGHT > 0
        self.kd_weight = cfg.CONT.DIST.KD_WEIGHT if self.use_kd else 0.

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight, "loss_kd": self.kd_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        self.criterion = KDSetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            old_classes=self.old_classes, use_kd=self.use_kd,
            uce=cfg.CONT.DIST.UCE, ukd=cfg.CONT.DIST.UKD, kd_deep=cfg.CONT.DIST.KD_DEEP,
            use_bkg=self.use_bg
        ).to(self.device)

    def __call__(self, data):

        model_out = self.model(data)
        outputs = model_out['outputs']

        if self.use_kd:
            model_out_old = self.model_old(data)
            outputs_old = model_out_old['outputs']
        else:
            outputs_old = None

        # prepare targets...
        if "instances" in data[0]:
            gt_instances = [x["instances"].to(self.device) for x in data]
            targets = MaskFormer.prepare_targets(gt_instances, model_out['shape'], per_pixel=False)
            if not self.use_bg:
                for tar in targets:
                    tar['labels'] -= 1
        else:
            targets = None

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets, outputs_old)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses
