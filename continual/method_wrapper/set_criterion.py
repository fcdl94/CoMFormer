import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.modeling.matcher import HungarianMatcher
from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from mask2former.modeling.criterion import calculate_uncertainty, dice_loss_jit, sigmoid_ce_loss_jit, SetCriterion


def focal_loss(inputs, targets, weights, alpha=1, gamma=2):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=255, weight=weights)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def unbiased_cross_entropy_loss(inputs, targets, weights, old_cl):
    outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
    den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax

    to_sum = torch.cat((inputs[:, -1:], inputs[:, 0:old_cl]), dim=1)
    outputs[:, -1] = torch.logsumexp(to_sum, dim=1) - den  # B, H, W       p(O)
    outputs[:, :-1] = inputs[:, :-1] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

    loss = F.nll_loss(outputs, targets, weight=weights)

    return loss


def knowledge_distillation_loss(inputs, targets,  weight_eos=1., alpha=1):
    # ToImprove: use only the logits when no_Class is not maximum (or reweight)
    # inputs = inputs.narrow(1, 0, targets.shape[1])
    outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
    outputs = torch.cat((outputs[:, :targets.shape[1]-1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    labels = torch.softmax(targets * alpha, dim=1)  # remmove no-class

    loss = - (outputs * labels)
    loss[:, -1] *= weight_eos
    loss = loss.mean(dim=1)

    return torch.mean(loss)


def unbiased_knowledge_distillation_loss(inputs, targets, weight_eos=1., alpha=1):
    # ToImprove: use only the logits when no_Class is not maximum (or reweight)
    targets = targets * alpha

    den = torch.logsumexp(inputs, dim=1)  # B, H, W
    outputs_no_bgk = inputs[:, :targets.shape[1]-1] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
    outputs_bkg = torch.logsumexp(inputs[:, targets.shape[1]-1:], dim=1) - den  # B, H, W
    labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

    # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
    loss = - (labels[:, -1] * outputs_bkg * weight_eos + (labels[:, :-1] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

    return torch.mean(loss)


class KDSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 old_classes=0, use_kd=False, uce=False, ukd=False, use_bkg=False, kd_deep=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio)
        self.old_classes = old_classes
        self.uce = uce and old_classes != 0
        self.ukd = ukd
        self.use_kd = use_kd
        self.kd_deep = kd_deep

        assert not (use_bkg and (uce or ukd)), "Using background mask is not supported with UCE or UKD distillation."

    def loss_labels(self, outputs, targets, indices, num_masks, outputs_old=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if self.uce:
            loss_ce = unbiased_cross_entropy_loss(src_logits.transpose(1, 2), target_classes,
                                                  self.empty_weight, self.old_classes)
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if outputs_old is not None:
            tar_logits = outputs_old["pred_logits"].float()
            if self.ukd:
                loss_kd = unbiased_knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                               weight_eos=self.eos_coef)
            else:
                loss_kd = knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                      weight_eos=self.eos_coef)
            losses["loss_kd"] = loss_kd

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, outputs_old=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks, outputs_old=None):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, outputs_old)

    def forward(self, outputs, targets, outputs_old=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             outputs_old:  dict of tensors by old model , see the output specification of the model for the format
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, outputs_old))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if outputs_old is not None and self.kd_deep:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks,
                                               outputs_old["aux_outputs"][i])
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses