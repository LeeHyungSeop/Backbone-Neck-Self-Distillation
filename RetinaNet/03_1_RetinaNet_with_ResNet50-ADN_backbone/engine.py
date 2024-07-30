import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import torch.nn as nn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
import numpy as np

# 2024.03.31 @hslee
# remove low quality matches for KL-Divergence Loss
# Because i believe that the base model does not need to learn the distribution of low-quality bounding boxes of the super model.
def remove_low_quality_matches(anchors, targets, remove_threshold):
    # type: (Tensor, List[Dict[str, Tensor]], float) -> list[list]
    # check the shape of targets
    for target in targets:
        print(f"target['boxes'].shape : {target['boxes'].shape}")
        
    save_idxs_list = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        save_idxs = []
        if targets_per_image["boxes"].numel() == 0:
            save_idxs_list.append([])
            continue
        
        # get IoU
        IoU = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
             
        # IoU : (num_targets, num_anchors)
        for i in range(IoU.shape[0]):
            """
            True means save
            False meanas remove
            """
            # set True if IoU > remove_threshold
            # print(f"IoU[i].shape : {IoU[i].shape}")
            IoU[i] = IoU[i] > remove_threshold
            
            # get idx of True
            save_idxs = save_idxs + torch.where(IoU[i])[0].tolist()
        save_idxs_list.append(list(set(save_idxs))) # remove duplicate
    return save_idxs_list # (batch size, save_idxs_per_image)     
        

def train_one_epoch(model, optimizer, data_loader, device, epoch, args, scaler=None, skip_cfg=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict, intermedia_features_base, foreground_idxs_list = model(images, targets, skip=skip_cfg)

        bbox_regression_loss_base = loss_dict['bbox_regression'][0]
        classification_loss_base = loss_dict['classification'][0]
        real_losses_base = bbox_regression_loss_base + classification_loss_base

        if not math.isfinite(real_losses_base):
            print(f"Loss is {real_losses_base}, stopping training")
            print(real_losses_base)
            sys.exit(1)

        if scaler is not None:
            scaler.scale(real_losses_base).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            real_losses_base.backward()
            
            # # 2024.04.23 @hslee : find_unused_parameters=True for only base model training
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"param : {name}, grad : {param.grad}, (parameters that did not participate in the forward and backward passes because of no require_grad=True or skip_cfg)")
            #     else :
            #         print(f"param : {name}, grad : {param.shape}, (parameters that participated in the forward and backward passes)")
            # print("----------------------------------------------------")

            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=real_losses_base)
        metric_logger.update(bbox=bbox_regression_loss_base)
        metric_logger.update(clas=classification_loss_base)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

    return metric_logger
      
# 2024.04.12 @hslee :
def get_good_feature_idx(foreground_idxs_list, range_setp_list):
    # print(f"len(foreground_idxs_list) : {len(foreground_idxs_list)}")
    # print(f"range_setp_list : {range_setp_list}")
    res = 0
    max = 0
    for i in range(1, len(range_setp_list)):
        temp = np.array(foreground_idxs_list)
        # temp 값이 range_setp_list[i-1] ~ range_setp_list[i] 사이에 있으면 1, 아니면 0
        temp = (temp >= range_setp_list[i-1]) & (temp < range_setp_list[i])
        count = np.count_nonzero(temp)
        # print(f"{i-1} count : {count}")
        if count > max:
            max = count
            res = i-1
    return res
# exp1 : maxforeground_feature_kd
def train_one_epoch_onebackward_exp1(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            
            range_step_list = []
            this_step = 0
            range_step_list.append(this_step)
            for i in range(len(intermedia_features_super)) :
                # print(f"intermedia_features_super[{i}].shape : {intermedia_features_super[i].shape}")
                this_step += (9 * intermedia_features_super[i].shape[2] * intermedia_features_super[i].shape[3])
                range_step_list.append(this_step)
            # range_step_list = [0, 136800, 171000, 179550, 188100, 190323]
            
            good_feature_idx = get_good_feature_idx(foreground_idxs_list, range_step_list)
            # print(f"good_feature_idx : {good_feature_idx}")
            
            intermedia_features_base = intermedia_features_base[good_feature_idx]
            intermedia_features_super = intermedia_features_super[good_feature_idx]
            
            T = 1
            avg_pool_super = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_super, (1, 1)))
            avg_pool_base = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_base, (1, 1)))
            kd_loss = criterion_kd(F.log_softmax(avg_pool_base/T, dim=1), F.softmax(avg_pool_super.clone().detach()/T, dim=1)) * (T*T)
            
            losses_base = real_losses_base + kd_loss
            
            loss = (alpha) * real_losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(losses_base = losses_base)
        metric_logger.update(kd_loss = kd_loss)
        
    return metric_logger

# exp2 : exclude minforeground_feature_kd
def get_bad_feature_idx(foreground_idxs_list, range_setp_list):
    # print(f"len(foreground_idxs_list) : {len(foreground_idxs_list)}")
    # print(f"range_setp_list : {range_setp_list}")
    res = 0
    min = 99999
    for i in range(1, len(range_setp_list)):
        temp = np.array(foreground_idxs_list)
        # temp 값이 range_setp_list[i-1] ~ range_setp_list[i] 사이에 있으면 1, 아니면 0
        temp = (temp >= range_setp_list[i-1]) & (temp < range_setp_list[i])
        count = np.count_nonzero(temp)
        # print(f"{i-1} count : {count}")
        if count < min:
            min = count
            res = i-1
    return res
def train_one_epoch_onebackward_exp2(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            
            range_step_list = []
            this_step = 0
            range_step_list.append(this_step)
            for i in range(len(intermedia_features_super)) :
                # print(f"intermedia_features_super[{i}].shape : {intermedia_features_super[i].shape}")
                this_step += (9 * intermedia_features_super[i].shape[2] * intermedia_features_super[i].shape[3])
                range_step_list.append(this_step)
            # range_step_list = [0, 136800, 171000, 179550, 188100, 190323]
            
            bad_feature_idx = get_bad_feature_idx(foreground_idxs_list, range_step_list)
    
            # delete bad_feature_idx
            intermedia_features_base.pop(bad_feature_idx)
            intermedia_features_super.pop(bad_feature_idx)
            
            kd_loss = 0
            T = 1
            for feature_base, feature_super in zip(intermedia_features_base, intermedia_features_super):
                avg_pool_super = torch.squeeze(nn.functional.adaptive_avg_pool2d(feature_super, (1, 1)))
                avg_pool_base = torch.squeeze(nn.functional.adaptive_avg_pool2d(feature_base, (1, 1)))
                kd_loss += criterion_kd(F.log_softmax(avg_pool_base/T, dim=1), F.softmax(avg_pool_super.clone().detach()/T, dim=1)) * (T*T)
            kd_loss /= (len(intermedia_features_base) - 1)
            
            losses_base = (real_losses_base + kd_loss)

            loss = (alpha) * real_losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(losses_base = losses_base)
        metric_logger.update(kd_loss = kd_loss)
        
    return metric_logger

# exp3 : exclude first feature_kd
def train_one_epoch_onebackward_exp3(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            kd_loss = 0
            T = 1
            for i, (feature_base, feature_super) in enumerate(zip(intermedia_features_base, intermedia_features_super)):
                if i == 0:
                    continue
                avg_pool_super = torch.squeeze(nn.functional.adaptive_avg_pool2d(feature_super, (1, 1)))
                avg_pool_base = torch.squeeze(nn.functional.adaptive_avg_pool2d(feature_base, (1, 1)))
                kd_loss += criterion_kd(F.log_softmax(avg_pool_base/T, dim=1), F.softmax(avg_pool_super.clone().detach()/T, dim=1)) * (T*T)
            kd_loss = kd_loss / (len(intermedia_features_base) - 1)
            
            losses_base = (real_losses_base + kd_loss)

            loss = (alpha) * real_losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # print grad
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(f"param : {name}, grad : {param.grad}, (parameters that did not participate in the forward and backward passes because of no require_grad=True or skip_cfg)")
                #     else :
                #         print(f"param : {name}, grad : {param.shape}, (parameters that participated in the forward and backward passes)")
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(losses_base = losses_base)
        metric_logger.update(kd_loss = kd_loss)
        
    return metric_logger

# exp4 : same with engine2
def train_one_epoch_onebackward_exp4(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
                
            feature_list_super = []
            for feature in intermedia_features_super:
                feature = torch.squeeze(torch.nn.functional.adaptive_avg_pool2d(feature,(1,1)))
                feature_list_super.append(feature)
            features_super = torch.cat(feature_list_super, dim=1)
            # print(f"features_super.shape : {features_super.shape}")
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            feature_list_base = []
            for feature in intermedia_features_base:
                feature = torch.squeeze(torch.nn.functional.adaptive_avg_pool2d(feature,(1,1)))
                feature_list_base.append(feature)
            features_base = torch.cat(feature_list_base, dim=1)
            # print(f"features_base.shape : {features_base.shape}")
            
            T = 1
            super_kd_loss = criterion_kd(F.log_softmax(features_super/T, dim=1), F.softmax(features_base.clone().detach()/T, dim=1)) * (T*T)
            base_kd_loss = criterion_kd(F.log_softmax(features_base/T, dim=1), F.softmax(features_super.clone().detach()/T, dim=1)) * (T*T)
            
            losses_super = beta * real_losses_super + (1-beta) * super_kd_loss
            losses_base = (1-beta) * real_losses_base + beta * base_kd_loss
            
            loss = (alpha) * losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss = loss)
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(super_kd_loss = super_kd_loss)
        metric_logger.update(base_kd_loss = base_kd_loss)
        metric_logger.update(losses_super = losses_super)
        metric_logger.update(losses_base = losses_base)
        
        
    return metric_logger

# exp5 : 
def train_one_epoch_onebackward_exp5(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            losses_super = real_losses_super
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            losses_base = real_losses_base
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            
            range_step_list = []
            this_step = 0
            range_step_list.append(this_step)
            for i in range(len(intermedia_features_super)) :
                # print(f"intermedia_features_super[{i}].shape : {intermedia_features_super[i].shape}")
                this_step += (9 * intermedia_features_super[i].shape[2] * intermedia_features_super[i].shape[3])
                range_step_list.append(this_step)
            # range_step_list = [0, 136800, 171000, 179550, 188100, 190323]
            
            good_feature_idx = get_good_feature_idx(foreground_idxs_list, range_step_list)
            # print(f"good_feature_idx : {good_feature_idx}")
            
            intermedia_features_base = intermedia_features_base[good_feature_idx]
            intermedia_features_super = intermedia_features_super[good_feature_idx]
            
            avg_pool_super = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_super, (1, 1)))
            avg_pool_base = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_base, (1, 1)))
            
            T = 1
            # if base model is better than super model, super model should learn from base model
            if real_losses_base > real_losses_super:
                # print("here1")
                kd_loss = criterion_kd(F.log_softmax(avg_pool_super/T, dim=1), F.softmax(avg_pool_base.clone().detach()/T, dim=1)) * (T*T)
                losses_super += kd_loss
            else :
                # print("here2")
                kd_loss = criterion_kd(F.log_softmax(avg_pool_base/T, dim=1), F.softmax(avg_pool_super.clone().detach()/T, dim=1)) * (T*T)
                losses_base += kd_loss
            
            
            loss = (alpha) * losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(losses_super = losses_super)
        metric_logger.update(losses_base = losses_base)
        metric_logger.update(kd_loss = kd_loss)
        
    return metric_logger

# exp6 : intermedia feature kd loss(exp1) + prediction kd loss
def train_one_epoch_onebackward_exp6(
    model, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args,  
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None
    ): 
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    header = f"Epoch: [{epoch}]"
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        alpha = args.subpath_alpha
        beta = args.beta
        
        # print(f"alpha : {alpha}, beta : {beta}")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # 1. forward pass for super_net
            loss_dict_super, intermedia_features_super, foreground_idxs_list = model(images, targets, skip=skip_cfg_supernet) # if training 
            # super_net loss
            real_losses_super = loss_dict_super['classification'][0] + loss_dict_super['bbox_regression'][0]
            bbox_pred_super = loss_dict_super['bbox_regression'][1]
            cls_pred_super = loss_dict_super['classification'][1]
            # print(f"bbox_pred_super.shape : {bbox_pred_super.shape}, cls_pred_super.shape : {cls_pred_super.shape}")
            if not math.isfinite(real_losses_super):
                print(f"real_loss_super is {real_losses_super}, stopping training")
                sys.exit(1)
        
            # 2. forward pass for base_net
            loss_dict_base, intermedia_features_base, _ = model(images, targets, skip=skip_cfg_basenet) # if training 
            # base_net loss
            real_losses_base = loss_dict_base['classification'][0] + loss_dict_base['bbox_regression'][0]
            bbox_pred_base = loss_dict_base['bbox_regression'][1]
            cls_pred_base = loss_dict_base['classification'][1]
            # print(f"bbox_pred_base.shape : {bbox_pred_base.shape}, cls_pred_base.shape : {cls_pred_base.shape}")
            if not math.isfinite(real_losses_base):
                print(f"real_loss_base is {real_losses_base}, stopping training")
                sys.exit(1)
            
            
            range_step_list = []
            this_step = 0
            range_step_list.append(this_step)
            for i in range(len(intermedia_features_super)) :
                # print(f"intermedia_features_super[{i}].shape : {intermedia_features_super[i].shape}")
                this_step += (9 * intermedia_features_super[i].shape[2] * intermedia_features_super[i].shape[3])
                range_step_list.append(this_step)
            # range_step_list = [0, 136800, 171000, 179550, 188100, 190323]
            
            good_feature_idx = get_good_feature_idx(foreground_idxs_list, range_step_list)
            # print(f"good_feature_idx : {good_feature_idx}")
            
            intermedia_features_base = intermedia_features_base[good_feature_idx]
            intermedia_features_super = intermedia_features_super[good_feature_idx]
            
            T = 1
            avg_pool_super = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_super, (1, 1)))
            avg_pool_base = torch.squeeze(nn.functional.adaptive_avg_pool2d(intermedia_features_base, (1, 1)))
            kd_loss = criterion_kd(F.log_softmax(avg_pool_base/T, dim=1), F.softmax(avg_pool_super.clone().detach()/T, dim=1)) * (T*T)
            
            losses_base = real_losses_base + kd_loss
            
            loss = (alpha) * real_losses_super + (1-alpha) * losses_base
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(real_losses_super = real_losses_super)
        metric_logger.update(real_losses_base = real_losses_base)
        metric_logger.update(losses_base = losses_base)
        metric_logger.update(kd_loss = kd_loss)
        
    return metric_logger



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, skip=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"[subnet]{skip} Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        
        # 2024.04.04 @hslee : add , features because customized retinanet model returns features
        outputs, _, _ = model(images, skip=skip)
        

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
