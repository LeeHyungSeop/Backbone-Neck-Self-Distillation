import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops, misc as misc_nn_ops, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.transforms._presets import ObjectDetection
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from .resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights
from ._utils import handle_legacy_interface, _ovewrite_named_param

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import _box_loss, overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet50_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform


__all__ = [
    "RetinaNet",
    "RetinaNet_ResNet50_FPN_Weights",
    "RetinaNet_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


def _v1_to_v2_weights(state_dict, prefix):
    for i in range(4):
        for type in ["weight", "bias"]:
            old_key = f"{prefix}conv.{2*i}.{type}"
            new_key = f"{prefix}conv.{i}.0.{type}"
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)


def _default_anchorgen():
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    def __init__(self, in_channels, num_anchors, num_classes, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer=norm_layer
        )
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors, norm_layer=norm_layer)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }, self.regression_head.foreground_idx_list

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), "bbox_regression": self.regression_head(x)}

class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        prior_probability=0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs["cls_logits"]
        
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        # 2024.03.23 @hslee
        # add cls_logits for KL-Divergence Loss
        loss = _sum(losses) / len(targets)
        return loss, cls_logits

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }
    
    
    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(misc_nn_ops.Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self._loss_type = "l1"


    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            _v1_to_v2_weights(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        self.foreground_idx_list = []
        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()
            
            # 2024.04.10 @hslee : check the number of foreground
            # print(f"num_foreground : {num_foreground}")
            self.foreground_idx_list = self.foreground_idx_list + foreground_idxs_per_image.tolist()
            
            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                _box_loss(
                    self._loss_type,
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_regression_per_image,
                )
                / max(1, num_foreground)
            )

        # 2024.03.23 @hslee
        loss = _sum(losses) / max(1, len(targets))
        return loss, bbox_regression, self.foreground_idx_list # add bbox_regression for KL-Divergence Loss

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        num_skippable_stages=None,
        **kwargs,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
            
        self.backbone = backbone
        self.num_skippable_stages = num_skippable_stages
        self.foreground_idx_list = []

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None instead of {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        if head is None:
            print(f"backbone.out_channels : {backbone.out_channels}")
            print(f"anchor_generator.num_anchors_per_location()[0] : {anchor_generator.num_anchors_per_location()[0]}")
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                # print(f"topk_idxs : {topk_idxs}")

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections


    # 2024.03.31 @hslee
    # remove low quality matches for KL-Divergence Loss
    # Because i believe that the base model does not need to learn the distribution of low-quality bounding boxes of the super model.
    def remove_low_quality_matches(self, pred_bbox, target_bbox, remove_threshold):
        # type: (Tensor, Tensor, float) -> list[int]
        iou = box_ops.box_iou(pred_bbox, target_bbox)
        matched_idx = torch.where(iou > remove_threshold)[0]
        return matched_idx
        

    # 2024.03.20 @hslee (add skip=None)
    # making sure all `forward` function outputs participate in calculating loss. 
    def forward(self, images, targets=None, skip=[True, True, True, True]):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = [] 
        # images : batch size
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # 2024.03.20 @hslee
        # transform the input to (800, 1216) 
        images, targets = self.transform(images, targets) 
        # print(f"images.tensors.shape : {images.tensors.shape}") # images.tensors.shape : torch.Size([16, 3, 800, 1216])
        # print(f"len(targets) : {len(targets)}") # 16
        # print(f"targets[0] : \
            # {targets[0]['boxes'].shape}, {targets[0]['labels'].shape}, {targets[0]['masks'].shape} \
        # ")
            # targets[0] : torch.Size([1, 4]), torch.Size([1]), torch.Size([1, 800, 1066])

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        
        # print(f"self.backbone : {self.backbone}") # BackboneWithADNFPN(
        # get the features from the backbone
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (32768x1 and 2048x1000)
        # [2024-03-22 15:16:58,517] torch.distributed.elastic.multiprocessing.api: \
            # [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 2323658) of binary: /home/hslee/anaconda3/envs/DL/bin/python
            # delete last layer of resnet50
        features = self.backbone(images.tensors, skip=skip)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())
        
        # 2024.04.04 @ hslee : get the features from the backbone for distillation between base and super model
        # print(f"len(features) : {len(features)}") # 5
        """
        len(features) : 5
            features[0].shape : torch.Size([4, 256, 100, 152])
            features[1].shape : torch.Size([4, 256, 50, 76])
            features[2].shape : torch.Size([4, 256, 25, 38])
            features[3].shape : torch.Size([4, 256, 25, 38])
            features[4].shape : torch.Size([4, 256, 13, 19])
            
            -> use these features for distillation between base and super model
        """
        

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)
        # print(f"head_outputs['cls_logits'].shape : {head_outputs['cls_logits'].shape}") 
            # (bs, 190323, 91)
        # print(f"head_outputs['bbox_regression'].shape : {head_outputs['bbox_regression'].shape}")
            # (bs, 190323, 4)
        
        # create the set of anchors
        anchors = self.anchor_generator(images, features)
        # print(f"len(anchors) : {len(anchors)}")
            # len(anchors) : {bs}


        # 2024.03.22 @hslee 
        # having trouble with losses, detections's return type is str in train_custom.py > train_one_epoch_twobackward()
        ## ----------------------------------------------------------------------------------------------------------------
        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses, self.foreground_idx_list = self.compute_loss(targets, head_outputs, anchors)
                # print(f"losses : {losses}")
        
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        
        # 2024.03.23 @hslee
        # In exp1, exp2 : use detections (bs, 190323, 4) for KL-Divergence Loss
        # return self.eager_outputs(losses, detections) 
            
        # 2024.04.04 @hslee
        # In exp3 ~     : use features for distillation between base and super model
        # because it's too many anchors to calculate KL-Divergence Loss in exp1, exp2
        # return features, not detections too for distillation between base and super model
        
        # 2024.04.11 @hslee
        # self.foreground_idx_list : list of foreground indices for each image
        # because it's too many anchors to calculate KL-Divergence Loss in exp1, exp2  
        return self.eager_outputs(losses, detections), features, self.foreground_idx_list
        # return self.eager_outputs(losses, detections)
        
        
        
    ## ----------------------------------------------------------------------------------------------------------------


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "min_size": (1, 1),
}


class RetinaNet_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 34014999,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#retinanet",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 36.4,
                }
            },
            "_ops": 151.54,
            "_file_size": 130.267,
            "_docs": """These weights were produced by following a similar training recipe as on the paper.""",
        },
    )
    DEFAULT = COCO_V1


class RetinaNet_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 38198935,
            "recipe": "https://github.com/pytorch/vision/pull/5756",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 41.5,
                }
            },
            "_ops": 152.238,
            "_file_size": 146.037,
            "_docs": """These weights were produced using an enhanced training recipe to boost the model accuracy.""",
        },
    )
    DEFAULT = COCO_V1


@register_model()
@handle_legacy_interface(
    weights=("pretrained", RetinaNet_ResNet50_FPN_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def retinanet_resnet50_adn_fpn(
    *,
    weights: Optional[RetinaNet_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    test_only=False,
    **kwargs: Any,
) -> RetinaNet:

    # 2024.03.20 @hslee
    weights = RetinaNet_ResNet50_FPN_Weights.verify(weights)
    
    # 2024.04.03 @hslee : now, --weights-backbone option is valid
    if test_only == False :
        weights_backbone = torch.load(weights_backbone)
    
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91
        
    # 2024.03.21 @hslee
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    # (default) norm_layer
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d
    
    # 2024.05.16 @hslee
    # this FrozenBatchNorm2d to flops calculation (Pytorch 151.54GFLOPs is calculated by excluding BN)
    # norm_layer = misc_nn_ops.FrozenBatchNorm2d
    # norm_layer = nn.BatchNorm2d
    
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)    
    num_skippable_stages = backbone.num_skippable_stages
    backbone = _resnet50_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    
    # anchor_generator = _default_anchorgen()
    # head = RetinaNetHead(
    #     backbone.out_channels,
    #     anchor_generator.num_anchors_per_location()[0],
    #     num_classes,
    #     norm_layer=partial(nn.GroupNorm, 32),
    # )
    # head.regression_head._loss_type = "giou"
    model = RetinaNet(backbone, num_classes, **kwargs, num_skippable_stages=num_skippable_stages)
    
    # print trainable layers
    for name, param in model.named_parameters():
        print(f"name : {name}, requires_grad : {param.requires_grad}")
        

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == RetinaNet_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model
