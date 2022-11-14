from .base import BaseDistillation
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.matcher import HungarianMatcher, SoftmaxMatcher
from .set_criterion import KDSetCriterion, SoftmaxKDSetCriterion
from .set_pseudo import PseudoSetCriterion
import torch
import torch.nn.functional as F
from continual.modeling.pod import func_pod_loss


def pod_loss(output, output_old):
    # fixme actually pod is computed BEFORE ReLU. Due to Detectron, it is hard to move after...
    input_feat = output['features']
    input_feat = [input_feat[key] for key in ['res2', 'res3', 'res4', 'res5']] + [output['outputs']['features']]

    old_feat = output_old['features']
    old_feat = [old_feat[key].detach() for key in ['res2', 'res3', 'res4', 'res5']] + [output_old['outputs']['features']]

    loss = {"loss_pod": func_pod_loss(input_feat, old_feat, scales=[1, 2, 4])}
    return loss


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
        self.no_object_weight = no_object_weight

        self.use_kd = cfg.CONT.TASK and cfg.CONT.DIST.KD_WEIGHT > 0
        self.kd_weight = cfg.CONT.DIST.KD_WEIGHT if self.use_kd else 0.
        self.pod_weight = cfg.CONT.DIST.POD_WEIGHT

        self.pseudolabeling = cfg.CONT.DIST.PSEUDO
        self.pseudo_type = cfg.CONT.DIST.PSEUDO_TYPE
        self.pseudo_thr = cfg.CONT.DIST.PSEUDO_THRESHOLD
        self.alpha = cfg.CONT.DIST.ALPHA
        self.pseudo_mask_threshold = 0.5
        self.iou_threshold = cfg.CONT.DIST.IOU_THRESHOLD

        self.softmask = cfg.MODEL.MASK_FORMER.SOFTMASK

        # building criterion
        if self.softmask:
            matcher = SoftmaxMatcher(
                cost_class=class_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        else:
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight, "loss_kd": self.kd_weight, "loss_mask_kd": cfg.CONT.DIST.MASK_KD,
                       "loss_pod": cfg.CONT.DIST.POD_WEIGHT * (self.new_classes / self.num_classes)**0.5}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        # if cfg.CONT.DIST.PSEUDO:
        #     self.criterion = PseudoSetCriterion(
        #         self.num_classes,
        #         matcher=matcher,
        #         weight_dict=weight_dict,
        #         eos_coef=no_object_weight,
        #         losses=losses,
        #         num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        #         oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
        #         importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        #         deep_sup=cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION,
        #         use_kd=self.use_kd, old_classes=self.old_classes, use_bkg=self.use_bg, alpha=cfg.CONT.DIST.ALPHA,
        #         weight_masks=(cfg.CONT.DIST.WEIGHT_MASK >= 0),
        #         focal=cfg.MODEL.MASK_FORMER.FOCAL,
        #     )
        # else:
        criterion = SoftmaxKDSetCriterion if self.softmask else KDSetCriterion
        self.criterion = criterion(
            self.num_classes,
            matcher=matcher,
            # Parameters for learning new classes
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            focal=cfg.MODEL.MASK_FORMER.FOCAL,
            focal_alpha=cfg.MODEL.MASK_FORMER.FOCAL_ALPHA, focal_gamma=cfg.MODEL.MASK_FORMER.FOCAL_GAMMA,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            deep_sup=cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION,
            # Parameters for not forget
            old_classes=self.old_classes, use_kd=self.use_kd, use_bkg=self.use_bg,
            uce=cfg.CONT.DIST.UCE, ukd=cfg.CONT.DIST.UKD, l2=cfg.CONT.DIST.L2, kd_deep=cfg.CONT.DIST.KD_DEEP,
            alpha=cfg.CONT.DIST.ALPHA, kd_reweight=cfg.CONT.DIST.KD_REW, mask_kd=cfg.CONT.DIST.MASK_KD,
        )
        self.criterion.to(self.device)

    def make_pseudolabels2(self, out, data, targets):
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQxK, BxQxHxW
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        for i in range(logits.shape[0]):  # iterate on batch size
            scores, labels = F.softmax(logits[i], dim=-1).max(-1)
            mask_pred = mask[i].sigmoid() if not self.softmask else mask[i].softmax(dim=0)

            keep = labels.ne(self.old_classes) & (scores > self.pseudo_thr)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_masks_bin = mask_pred[keep].clone()

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            tar = targets[i]
            gt_pixels = tar['masks'].sum(dim=0).bool()  # H,W
            keep2 = torch.zeros(len(cur_masks)).bool()

            if cur_masks.shape[0] > 0:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)  # REMOVE GT
                cur_mask_ids[gt_pixels] = -1

                for k in range(cur_classes.shape[0]):
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    x_mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and x_mask.sum().item() > 0:
                        if mask_area / original_area > 0.5:
                            keep2[k] = 1
                            cur_masks_bin[k] = x_mask

            if keep2.sum() > 0:
                # probs = cur_scores[keep2]
                # new_zeros = torch.zeros((probs.shape[0], self.num_classes-self.old_classes),
                #                         device=probs.device, dtype=probs.dtype)
                # pseudo_probs = torch.cat((probs[:, :-1], new_zeros, probs[:, -1:]), dim=1)
                # pseudo_raw_mask = cur_masks[keep_2]
                pseudo_lab = cur_classes[keep2]
                pseudo_mask = cur_masks_bin[keep2].bool()

                tar['masks'] = torch.cat((tar['masks'], pseudo_mask), dim=0)
                tar['labels'] = torch.cat((tar['labels'], pseudo_lab), dim=0)
                # tar['probs'] = F.one_hot(tar['labels'], self.num_classes + 1).to(dtype=torch.float)
                # tar['probs'] = torch.cat((tar['probs'], pseudo_probs), dim=0)
                # tar['scores'] = tar['masks'].float()
                # tar['scores'] = torch.cat((tar['scores'], pseudo_raw_mask), dim=0)

        return targets

    @staticmethod
    def compute_similarity(pseudo, gt):
        with torch.no_grad():
            pseudo = pseudo.flatten(1).float()  # N x HW
            gt = gt.flatten(1).float()  # M x HW
            numerator = 2 * torch.einsum("nc,mc->nm", pseudo, gt)
            denominator = pseudo.sum(-1)[:, None] + gt.sum(-1)[None, :]
            dice = (numerator) / (denominator + 1)
        return dice

    def make_pseudolabels(self, out, data, targets):
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQxK, BxQxHxW
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        for i in range(logits.shape[0]):  # iterate on batch size
            tar = targets[i]
            ist = {}
            l = (logits[i] * self.alpha).softmax(dim=-1)
            m = mask[i].sigmoid() if not self.softmask else mask[i].softmax(dim=0)   # Get logits and scores
            scores, classes = l.max(dim=-1)  # compute the max and the argmax of logits -> assign class to masks
            m_bin = (m > self.pseudo_mask_threshold).float()
            # q = m.shape[0]
            # m_scores = (m * m_bin).view(q, -1).sum(dim=1) / (m_bin.view(q, -1).sum(dim=1) + 1e-4)
            keep = (classes != self.old_classes) & (scores > self.pseudo_thr)

            gt_pixels = tar['masks'].sum(dim=0, keepdim=True)  # 1,H,W
            iou = MaskFormerDistillation.compute_similarity(m_bin[keep], gt_pixels)
            keep_2 = (iou > self.iou_threshold).sum(dim=1) == 0
            ist['n_mask'] = keep_2.sum()
            if ist['n_mask'] > 0:
                # probs = l[keep][keep_2]
                # new_zeros = torch.zeros((probs.shape[0], self.num_classes-self.old_classes),
                #                         device=probs.device, dtype=probs.dtype)
                # p_distr = torch.cat((probs[:, :-1], new_zeros, probs[:, -1:]), dim=1)  # add padding for new classes
                ist['labels'] = classes[keep][keep_2]  # labels are the classes
                ist["masks"] = m_bin[keep][keep_2].bool()  # * (1-gt_pixels)).bool()
                # ist['probs'] = p_distr  # class confidence * factor
                # ist['scores'] = m_scores[keep][keep_2]  # mask confidence * factor

                # PREPARE FINAL TARGETS
                # tar['scores'] = torch.ones(tar['labels'].shape[0], dtype=torch.float, device=tar['labels'].device)
                # tar['probs'] = F.one_hot(tar['labels'], self.num_classes + 1).to(dtype=torch.float)
                tar['masks'] = torch.cat((tar['masks'], ist['masks']), dim=0)
                tar['labels'] = torch.cat((tar['labels'], ist['labels']), dim=0)
                # tar['probs'] = torch.cat((tar['probs'], ist['probs']), dim=0)
                # tar['scores'] = torch.cat((tar['scores'], ist['scores']), dim=0)

        return targets  # we modify in place, but return it anyway

    def __call__(self, data):
        model_out = self.model(data)
        outputs = model_out['outputs']

        model_out_old = self.model_old(data) if self.use_kd or self.pseudolabeling else None
        outputs_old = model_out_old['outputs'] if self.use_kd and model_out_old is not None else None

        # prepare targets...
        if "instances" in data[0]:
            gt_instances = [x["instances"].to(self.device) for x in data]
            targets = MaskFormer.prepare_targets(gt_instances, model_out['shape'], per_pixel=False)

            # Labels assume that background is class 0, remove it.
            if not self.use_bg:
                for tar in targets:
                    tar['labels'] -= 1

            # Pseudo-labeling algorithm
            if self.pseudolabeling:
                if self.pseudo_type == 0:
                    targets = self.make_pseudolabels(model_out_old, data, targets)
                else:
                    targets = self.make_pseudolabels2(model_out_old, data, targets)

        else:
            targets = None

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets, outputs_old)

        if self.pod_weight > 0:
            losses.update(pod_loss(model_out, model_out_old))

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses
