"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 
import torch.nn.functional as F 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

def loss(S, T, gamma = 1):
    # N : number of elements in each feature map
    N = S.shape[1] * S.shape[2] * S.shape[3] 
    # (bs, C, H, W) -> (bs, H, W, C)
    S = S.permute(0, 2, 3, 1)
    T = T.permute(0, 2, 3, 1)
    loss = (gamma / N) * torch.sum((S - T) ** 2)
    return loss


def total_loss(S_s, T_s, S_m, T_m, S_l, T_l, gamma):
    L_s = loss(S_s, T_s, gamma)
    L_m = loss(S_m, T_m)
    L_l = loss(S_l, T_l)
    return L_s + L_m + L_l


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            
            wNeck = True
            outputs_w_neck = model(samples, targets, wNeck=wNeck)
            neck_outs = outputs_w_neck.pop('neck_outs')
            backbone_outs = outputs_w_neck.pop('backbone_outs')
            # for i, out in enumerate(neck_outs):
            #     print(f"\tneck_out[{i}] : {out.shape}")
            # for i, out in enumerate(backbone_outs):
            #     print(f"\tbackbone_out[{i}] : {out.shape}")
            
            # wNeck = False
            # outputs_wo_neck = model(samples, targets, wNeck=wNeck)
            
            
            
            # 2024.07.25 @hslee : criterion -> src/zoo/rtdetr/rtdetr_criterion.py
            
            """
            # outputs key
                '''
                [Output Key] : pred_logits
                [Output Key] : pred_boxes
                [Output Key] : aux_outputs
                [Output Key] : dn_aux_outputs
                [Output Key] : dn_meta
                '''
            
            # print(f"[Final Output]")
            for key, value in outputs.items():
                
                if key == 'pred_logits':
                    print(f"\t{key} : {value.shape}")
                    # pred_logits : torch.Size([4, 300, 80])
                    
                elif key == 'pred_boxes':
                    print(f"\t{key} : {value.shape}")
                    # pred_boxes : torch.Size([4, 300, 4])
                    
                elif key == 'aux_outputs' or key == 'dn_aux_outputs': # key : list(key, value)
                    print(f"\t{key} : ")
                    for i in range(len(value)):
                        for k, v in value[i].items():
                            print(f"\t\t{key}[{i}][{k}] : {v.shape}")
                            # aux_outputs : 
                                # aux_outputs[0][pred_logits] : torch.Size([4, 300, 80])
                                # aux_outputs[0][pred_boxes] : torch.Size([4, 300, 4])
                                # ...
                                # aux_outputs[5][pred_logits] : torch.Size([4, 300, 80])
                                # aux_outputs[5][pred_boxes] : torch.Size([4, 300, 4])
                            # dn_aux_outputs : 
                                # dn_aux_outputs[0][pred_logits] : torch.Size([4, 200, 80])
                                # dn_aux_outputs[0][pred_boxes] : torch.Size([4, 200, 4])
                                # ...
                                # dn_aux_outputs[5][pred_logits] : torch.Size([4, 200, 80])
                                # dn_aux_outputs[5][pred_boxes] : torch.Size([4, 200, 4])
            
                elif key == 'dn_meta' : # key : (key, value)
                    print(f"\t{key} : ")
                    for k, v in value.items():
                        print(f"\t\t{key}[{k}] : {v}")
                        # dn_meta : 
                            # dn_meta[dn_positive_idx] : tuple data type -> (tensor, tensor, tensor, tensor)
                            # dn_meta[dn_num_group] : scalar
                            # dn_meta[dn_num_split] : [scalar, scalar]
            """
            
            loss_w_neck_dict = criterion(outputs_w_neck, targets)
            # loss_wo_neck_dict = criterion(outputs_wo_neck, targets)
            
            loss_w_neck = sum(loss_w_neck_dict.values())
            # loss_wo_neck = sum(loss_wo_neck_dict.values())
            
            
            optimizer.zero_grad()
            loss_nb_kd = 0
            
            # # 1. KLDiv
            # for i in range(3):
            #     loss_nb_kd += F.kl_div(F.log_softmax(neck_outs[i], dim=1), F.softmax(backbone_outs[i], dim=1))
            # loss_nb_kd /= 3
            

            # 2. MSSD
            gamma = 0.05
            loss_nb_kd = total_loss(backbone_outs[0], neck_outs[0], backbone_outs[1], neck_outs[1], backbone_outs[2], neck_outs[2], gamma)
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
            # jointly train with neck and wo_neck
            # alpha = 0.5
            # loss = alpha * loss_w_neck + (1 - alpha) * loss_wo_neck + loss_nb_kd
            loss = loss_w_neck + loss_nb_kd
            loss.backward()
            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_w_neck_reduced = reduce_dict(loss_w_neck_dict)
        # loss_dict_wo_neck_reduced = reduce_dict(loss_wo_neck_dict)
        
        loss_value_w_neck = sum(loss_dict_w_neck_reduced.values())
        # loss_value_wo_neck = sum(loss_dict_wo_neck_reduced.values())
        # loss_value = alpha * loss_value_w_neck + (1 - alpha) * loss_value_wo_neck + loss_nb_kd
        loss_value = loss_value_w_neck + loss_nb_kd

        if not math.isfinite(loss_value_w_neck):
            print("Loss is {}, stopping training".format(loss_value_w_neck))
            print(loss_dict_w_neck_reduced)
            sys.exit(1)
        # if not math.isfinite(loss_value_wo_neck):
        #     print("Loss is {}, stopping training".format(loss_value_wo_neck))
        #     print(loss_dict_wo_neck_reduced)
        #     sys.exit(1)

        
        metric_logger.update(loss=loss_value)
        metric_logger.update(KD=loss_nb_kd)
        metric_logger.update(loss_w_neck=loss_value_w_neck, **loss_dict_w_neck_reduced)
        # metric_logger.update(loss_wo_neck=loss_value_wo_neck, **loss_dict_wo_neck_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, wNeck):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples, wNeck=wNeck)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator



