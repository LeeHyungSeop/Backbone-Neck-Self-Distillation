"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import time
import torch
import torch.utils.data  
import torchvision

import torch.cuda.profiler as profiler
from ptflops import get_model_complexity_info

from thop import profile
from thop import clever_format



import presets
import utils
from torch import nn
from coco_utils import get_coco
from engine import evaluate, \
    train_one_epoch, train_one_epoch_onebackward_exp1, train_one_epoch_onebackward_exp2, \
    train_one_epoch_onebackward_exp3, train_one_epoch_onebackward_exp4, train_one_epoch_onebackward_exp5, \
    train_one_epoch_onebackward_exp6 \

from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
import os,sys
from transforms import SimpleCopyPaste
import models
os.environ["OMP_NUM_THREADS"] = '8'

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))

def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )
    parser.add_argument("--weights-path", default=None, help='path to weights file')
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    
    parser.add_argument("--model", default="retinanet_resnet50_adn_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    
    # parser.add_argument(
    #     "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    # )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--subpath-alpha", default=0.5, type=float, help="sub-paths distillation alpha (default: 0.5)")
    parser.add_argument("--beta", default=0.9, type=float, help="weight to balance the base model loss and kd loss")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser

def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    # 2024.03.20 hslee
    # have to modify this part to use new model --------------------------------------------------------------------------------------
    # if args.model not in ("retinanet_resnet50_adn_fpn", "swin_t", "vit_b_16", "vit_b_32", "efficientnet_v2_s", "efficientnet_b2"):
    if args.model not in models.__dict__.keys():
        print(f"{args.model} is not supported")
        sys.exit()
    if args.weights_backbone is not None and args.test_only == False :
        print(f"Loading weights from {args.weights_backbone}")
        model = models.__dict__[args.model](weights_backbone = args.weights_backbone)
        model.to(device)
    else :
        model = models.__dict__[args.model](test_only = args.test_only, weights_backbone = args.weights_backbone)
        model.to(device)
    # --------------------------------------------------------------------------------------------------------------------------------      

    # 2024.03.21 @hslee
    # 2024.04.12 @hslee : log_target = True (False is default)
    criterion_kd = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    print(f"model : {model}")
    # make requires_grad=False to layer 1
    for name, param in model.named_parameters():
        if 'layer1' in name :
            param.requires_grad = False
    
    # print the name of parameters, and the number of parameters
    skip_bn_sum = 0
    layer1_sum = 0
    down_sum =0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.numel(), "requires_grad=False")
        else :
            print(name, param.numel(), "requires_grad=True")
        if 'bn' in name :
            skip_bn_sum += param.numel()
        if 'downsample.1' in name : 
            down_sum += param.numel()
    numParams = sum(p.numel() for p in model.parameters())
    print(f"#Params : ",numParams)
    # when test time, frozenBN is used so that the number of parameters is reduced
    print(f"#Params(without downsample, bn): {numParams - down_sum - skip_bn_sum}")
    


    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 2024.03.20 @hslee
    num_skippable_stages = model_without_ddp.num_skippable_stages
    print(f"num_skippable_stages : {num_skippable_stages}")
    skip_cfg_supernet = [False for _ in range(num_skippable_stages)]
    skip_cfg_basenet = [True for _ in range(num_skippable_stages)]
    print(f"skip_cfg_supernet : {skip_cfg_supernet}")
    print(f"skip_cfg_basenet : {skip_cfg_basenet}")
    
    if args.test_only :
        torch.backends.cudnn.deterministic = True
        checkpoint = torch.load(args.weights_path)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.to(args.device)
    
        # with torch.cuda.device(0) : 
        #     flops, params = get_model_complexity_info(model_without_ddp, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True, backend='pytorch')
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
        input = torch.randn(1, 3, 224, 224).to(args.device)
        flops, params = profile(model_without_ddp, inputs=(input, ))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"flops: {flops}, params: {params}")

        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
    
        # 2024.03.20 @hslee
        # train_one_epoch(model, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_supernet)
        # train_one_epoch(model, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet)
        # train_one_epoch_onebackward_exp1(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)
        # train_one_epoch_onebackward_exp2(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)
        # train_one_epoch_onebackward_exp3(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)
        # train_one_epoch_onebackward_exp4(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)
        # train_one_epoch_onebackward_exp5(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)
        # train_one_epoch_onebackward_exp6(model, criterion_kd, optimizer, data_loader, device, epoch, args, scaler, skip_cfg_basenet, skip_cfg_supernet)

       
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        
        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device, skip=skip_cfg_supernet)
        evaluate(model, data_loader_test, device=device, skip=skip_cfg_basenet)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)


# ampere
'''
    lr : 0.02 / 8 * #NGPN
        if #NGPN = 4, lr = 0.02 / 8 * 4 = 0.01

    (train)
    torchrun --nproc_per_node=4 train_custom.py --dataset coco --data-path=/media/data/coco \
    --model retinanet_resnet50_adn_fpn --epochs 26 \
    --batch-size 4 --workers 8 --lr-steps 16 22 \
    --world-size 4 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --subpath-alpha 0.5 \
    --weights-backbone /home/hslee/INU_RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs2/super_model.txt
    
    (customized recipe)
    torchrun --nproc_per_node=4 train_custom.py \
    --dataset coco --data-path=/media/data/coco \
    --model retinanet_resnet50_adn_fpn --epochs 13 \
    --batch-size 4 --workers 8 --lr-steps 8 11 \
    --world-size 4 --world-size 4 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --subpath-alpha 0.5 --beta 0.9 \
    --weights-backbone /home/hslee/INU_RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs/exp6_test.txt
    
    (test)
    torchrun --nproc_per_node=1 train_custom_flops.py --dataset coco --data-path=/media/data/coco \
    --model retinanet_resnet50_adn_fpn \
    --batch-size 4 --workers 8 \
    --test-only --weights-path /home/hslee/INU_RISE/03_1_RetinaNet_with_ResNet50-ADN_backbone/pretrained/exp1_goodF_alpha05_noBeta_pyTorchSchedule_model_23.pth \
    2>&1 | tee ./logs/flops_super.txt
    
    (baseline)
    torchrun --nproc_per_node=4 train_custom.py \
    --dataset coco --data-path=/media/data/coco \
    --model retinanet_resnet50_fpn --epochs 26 \
    --batch-size 4 --workers 8 --lr-steps 16 22 \
    --world-size 4 --world-size 4 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --weights-backbone ResNet50_Weights.IMAGENET1K_V1 \
    2>&1 | tee ./logs/basemodel_baseline.txt
'''

# Desktop
'''
    lr : 0.02 / 8 * #NGPN
        if #NGPN = 1, lr = 0.02 / 8 * 4 = 0.01
        
    torchrun --nproc_per_node=1 train_custom.py --dataset coco --data-path=/home/hslee/Desktop/Datasets/COCO \
    --model retinanet_resnet50_adn_fpn --epochs 26 \
    --batch-size 8 --workers 8 --lr-steps 16 22 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --subpath-alpha 0.5 --clip-grad-norm 1.0 \
    --weights-backbone /home/hslee/Desktop/Embedded_AI/INU_4-1/RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs2/base_model_custom_recipe.txt
    
    (customized recipe)
    torchrun --nproc_per_node=1 train_custom.py --dataset coco --data-path=/home/hslee/Desktop/Datasets/COCO \
    --model retinanet_resnet50_adn_fpn --epochs 16 \
    --batch-size 8 --workers 8 --lr-steps 13 15 \
    --aspect-ratio-group-factor 3 --lr 0.01 \
    --subpath-alpha 0.5 \
    --weights-backbone /home/hslee/Desktop/Embedded_AI/INU_4-1/RISE/02_AdaptiveDepthNetwork/pretrained/resnet50_adn_model_145.pth \
    2>&1 | tee ./logs2/super_model_custom_recipe_test.txt
'''


'''


'''