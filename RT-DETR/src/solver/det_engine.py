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
import torch.nn as nn
import torch.nn.functional as F 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


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
            
            # 2024.08.01 @hslee KL-Div (forward-backward, forward-backward, update)
            
            optimizer.zero_grad()
            alpha = 0.5 # wneck : weight to alpha, w/o neck : weight to (1 - alpha)
            T = 4.0
             
            wNeck = True
            outputs_w_neck = model(samples, targets, wNeck=wNeck)
            backbone_outs = outputs_w_neck.pop('backbone_outs') 
            neck_outs = outputs_w_neck.pop('neck_outs')
            
            loss_w_neck_dict = criterion(outputs_w_neck, targets)
            loss_w_neck = sum(loss_w_neck_dict.values())
            loss_w_neck = alpha * loss_w_neck
            
            # loss_w_neck.backward(retain_graph=True)
            
            
            wNeck = False
            outputs_wo_neck = model(samples, targets, wNeck=wNeck)
            loss_wo_neck_dict = criterion(outputs_wo_neck, targets)
            loss_wo_neck = sum(loss_wo_neck_dict.values())
            loss_wo_neck = (1 - alpha) * loss_wo_neck
            
            
            ## KL-Div
            kl_div = nn.KLDivLoss(reduction='batchmean')
            loss_nb_kd = 0
            loss_bn_kd = 0
            for i in range(len(neck_outs)): # feature map (bs, c, h, w)
                # N -> B
                teacher = F.softmax(neck_outs[i] / T, dim=1).clone().detach()
                student = F.log_softmax(backbone_outs[i] / T, dim=1)
                loss_nb_kd += kl_div(student, teacher) * (T * T)
                # B -> N
                teacher = F.softmax(backbone_outs[i] / T, dim=1).clone().detach()
                student = F.log_softmax(neck_outs[i] / T, dim=1)
                loss_bn_kd += kl_div(student, teacher) * (T * T)
            loss_nb_kd /= len(neck_outs)
            loss_bn_kd /= len(neck_outs)
            
            # kl_div = nn.KLDivLoss(reduction='batchmean')
            # loss_nb_kd = 0
            # for i in range(len(neck_outs)): # feature map (bs, c, h, w)
            #     student = F.log_softmax(backbone_outs[i] / T, dim=1)
            #     teacher = F.softmax(neck_outs[i] / T, dim=1).clone().detach()
                
            #     # average pooling
            #     student = F.adaptive_avg_pool2d(student, (1, 1)).squeeze() # (bs, c)
            #     teacher = F.adaptive_avg_pool2d(teacher, (1, 1)).squeeze() # (bs, c)
                
            #     loss_nb_kd += kl_div(student, teacher) * (T * T)
            # loss_nb_kd /= len(neck_outs)
                
            
            # loss_wo_neck = loss_wo_neck + 0.001 *  loss_nb_kd
            # loss_wo_neck.backward()
            
            # collaborative learning(cl) between backbone and neck
            loss_cl = loss_nb_kd + loss_bn_kd
            loss = loss_w_neck + loss_wo_neck + loss_nb_kd + loss_cl
            loss.backward()
            
            optimizer.step()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_w_neck_reduced = reduce_dict(loss_w_neck_dict)
        loss_dict_wo_neck_reduced = reduce_dict(loss_wo_neck_dict)
        
        loss_value_w_neck = sum(loss_dict_w_neck_reduced.values())
        loss_value_wo_neck = sum(loss_dict_wo_neck_reduced.values())
        loss_value = alpha * loss_value_w_neck + (1 - alpha) * loss_value_wo_neck + loss_nb_kd + loss_bn_kd
        # loss_value = alpha * loss_value_w_neck + (1 - alpha) * loss_value_wo_neck

        if not math.isfinite(loss_value_w_neck):
            print("Loss is {}, stopping training".format(loss_value_w_neck))
            print(loss_dict_w_neck_reduced)
            sys.exit(1)
        if not math.isfinite(loss_value_wo_neck):
            print("Loss is {}, stopping training".format(loss_value_wo_neck))
            print(loss_dict_wo_neck_reduced)
            sys.exit(1)

        
        metric_logger.update(loss=loss_value)
        # 2024.08.03 @hslee No KD Loss
        metric_logger.update(nb_kd=loss_nb_kd)
        metric_logger.update(bn_kd=loss_bn_kd)
        metric_logger.update(loss_w_neck=loss_value_w_neck, **loss_dict_w_neck_reduced)
        metric_logger.update(loss_wo_neck=loss_value_wo_neck, **loss_dict_wo_neck_reduced)
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



