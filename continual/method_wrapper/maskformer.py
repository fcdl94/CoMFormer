from .base import BaseDistillation
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.matcher import HungarianMatcher
from .set_criterion import KDSetCriterion
from .set_pseudo import PseudoSetCriterion
import torch
import torch.nn.functional as F


def knowledge_distillation_loss(inputs, targets,  no_object_weight=0.1, alpha=1., use_new=False):
    # ToImprove: use only the logits when no_Class is not maximum (or reweight)
    if use_new:
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
        outputs = torch.cat((outputs[:, :targets.shape[1]-1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    else:
        inputs = torch.cat((inputs[:, :targets.shape[1]-1], inputs[:, -1:]), dim=1)  # only old classes or EOS
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
    labels = torch.softmax(targets * alpha, dim=1)  # remove no-class

    loss = - (outputs * labels)  # B, K, Q
    loss[:, -1] = loss[:, -1] * no_object_weight
    # make weighted average of the queries for each image
    loss = loss.sum(dim=1)

    return torch.mean(loss)


class MaskFormerDistillation(BaseDistillation):
    def __init__(self, cfg, model, model_old):
        super().__init__(cfg, model, model_old)
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        cx = cfg.MODEL.MASK_FORMER
        class_weight = cx.CLASS_WEIGHT
        dice_weight = cx.DICE_WEIGHT
        mask_weight = cx.MASK_WEIGHT

        self.use_kd = cfg.CONT.TASK and cfg.CONT.DIST.KD_WEIGHT > 0
        self.kd_weight = cfg.CONT.DIST.KD_WEIGHT if self.use_kd else 0.
        self.pseudolabeling = cfg.CONT.DIST.PSEUDO
        self.alpha = cfg.CONT.DIST.ALPHA
        self.sanity = cfg.CONT.DIST.SANITY
        self.pseudo_mask_weight = cfg.CONT.DIST.WEIGHT_MASK
        self.inc_query = cfg.CONT.INC_QUERY
        self.first_new_query = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES * self.old_classes if cfg.CONT.INC_QUERY else 0
        self.no_object_weight = no_object_weight

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight if cx.CLASS_WEIGHT_MATCH < 0 else cx.CLASS_WEIGHT_MATCH,
            cost_mask=mask_weight if cx.MASK_WEIGHT_MATCH < 0 else cx.MASK_WEIGHT_MATCH,
            cost_dice=dice_weight if cx.DICE_WEIGHT_MATCH < 0 else cx.DICE_WEIGHT_MATCH,
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

        if cfg.CONT.DIST.PSEUDO:
            self.criterion = PseudoSetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                use_kd=self.use_kd, old_classes=self.old_classes, use_bkg=self.use_bg, alpha=cfg.CONT.DIST.ALPHA,
                weight_masks=(cfg.CONT.DIST.WEIGHT_MASK >= 0)
            )
        else:
            self.criterion = KDSetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                old_classes=self.old_classes, use_kd=self.use_kd, use_bkg=self.use_bg,
                uce=cfg.CONT.DIST.UCE, ukd=cfg.CONT.DIST.UKD, kd_deep=cfg.CONT.DIST.KD_DEEP,
                eos_pow=cfg.CONT.DIST.EOS_POW, alpha=cfg.CONT.DIST.ALPHA,
                ce_only_new=cfg.CONT.DIST.CE_NEW, kd_use_novel=cfg.CONT.DIST.USE_NEW,
            )
        self.criterion.to(self.device)

    def make_pseudolabels(self, out, data):
        res = []
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQxK, BxQxHxW
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        for i in range(logits.shape[0]):  # iterate on batch size
            ist = {}
            l, m = logits[i], mask[i]
            classes = l.argmax(dim=-1)
            keep = (classes != self.old_classes)
            if keep.sum() > 0:
                probs = (l[keep] * self.alpha).softmax(dim=1)
                q, _ = probs.shape
                new_zeros = torch.zeros((q, self.num_classes-self.old_classes), device=probs.device, dtype=probs.dtype)
                p_distr = torch.cat((probs[:, :-1], new_zeros, probs[:, -1:]), dim=1)
                scores, _ = probs.max(dim=-1)
                ist['labels'] = classes[keep]
                ist['scores'] = scores * self.pseudo_mask_weight
                ist['probs'] = p_distr * self.sanity
                ist["masks"] = (m[keep].sigmoid() > 0.5)

            ist['n_mask'] = keep.sum().item()
            res.append(ist)

        return res  # this should be a list of instances

    def cat_targets(self, targets, pseudolabels):
        # targets and pseudolabels are two lists of dict containing labels, masks, [scores]
        for tar, pseudo in zip(targets, pseudolabels):
            tar['scores'] = torch.ones(tar['labels'].shape[0], dtype=torch.float, device=tar['labels'].device)
            tar['probs'] = F.one_hot(tar['labels'], self.num_classes + 1)

            if pseudo['n_mask'] > 0:
                tar['probs'] = torch.cat((tar['probs'], pseudo['probs']), dim=0)
                tar['labels'] = torch.cat((tar['labels'], pseudo['labels']), dim=0)
                tar['masks'] = torch.cat((tar['masks'], pseudo['masks']), dim=0)
                tar['scores'] = torch.cat((tar['scores'], pseudo['scores']), dim=0)
        return targets  # we modify in place, but return it anyway

    def __call__(self, data):

        model_out = self.model(data)
        outputs = model_out['outputs']

        model_out_old = self.model_old(data) if self.use_kd or self.pseudolabeling else None
        outputs_old = model_out_old['outputs'] if self.use_kd and model_out_old is not None else None

        if self.inc_query and self.kd_weight > 0:
            kd_loss = knowledge_distillation_loss(outputs['pred_logits'][:, :self.first_new_query].transpose(1, 2),
                                                  outputs_old['pred_logits'].transpose(1, 2),
                                                  use_new=True, no_object_weight=self.no_object_weight)
            outputs_old = None

        if self.inc_query:
            outputs['pred_logits'] = outputs['pred_logits'][:, self.first_new_query:]
            outputs['pred_masks'] = outputs['pred_masks'][:, self.first_new_query:]
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    outputs["aux_outputs"][i]['pred_masks'] = outputs["aux_outputs"][i]['pred_masks'][:,
                                                              self.first_new_query:]
                    outputs["aux_outputs"][i]['pred_logits'] = outputs["aux_outputs"][i]['pred_logits'][:,
                                                               self.first_new_query:]

        # prepare targets...
        if "instances" in data[0]:
            gt_instances = [x["instances"].to(self.device) for x in data]
            targets = MaskFormer.prepare_targets(gt_instances, model_out['shape'], per_pixel=False)

            if not self.use_bg:
                for tar in targets:
                    tar['labels'] -= 1

            if self.pseudolabeling:
                pseudolabels = self.make_pseudolabels(model_out_old, data)
                targets = self.cat_targets(targets, pseudolabels)
                del pseudolabels

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

        if self.inc_query and self.kd_weight > 0:
            losses['loss_kd'] = self.kd_weight * kd_loss
        return losses
