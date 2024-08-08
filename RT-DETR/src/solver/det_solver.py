'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

from fvcore.nn import FlopCountAnalysis, flop_count_table

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import requests
import torchvision.transforms as transforms
from PIL import Image


COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # 2024.07.25 @hslee : original model training (woNeck = False)
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            
            # 2024.07.25 @hslee : without neck model evaluation
            wNeck = True
            print(f"wNeck : {wNeck}")
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir, wNeck=wNeck
            )
            wNeck = False
            print(f"wNeck : {wNeck}")
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir, wNeck=wNeck
            )
            
            # TODO 
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        # 2024.08.07 @hslee : Visualization of Class Activation Map

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        
        # ------------------------------------------------------------------------------
        # # 2024.07.25 @hslee : original model evaluation
        # wNeck = True
        # print(f"(evaluate) wNeck : {wNeck}")
        # test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
        #         self.val_dataloader, base_ds, self.device, self.output_dir, wNeck=wNeck)
                
        # if self.output_dir: 
        #     dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        # ------------------------------------------------------------------------------
            
        image_url =  "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
        img = np.array(Image.open(requests.get(image_url, stream=True).raw))
        img = cv2.resize(img, (640, 640))
        cv2.imwrite("input_image.jpg", img)
        rgb_img = img.copy()
        img = np.float32(img) / 255
        transform = transforms.ToTensor()
        tensor = transform(img).unsqueeze(0)
        # save input image
            
        module.eval()
        module.cpu()
        # check the attribute of the module
        print(f"module : {module}")
        # print(module.backbone.res_layers._modules['1'].blocks._modules['3'].branch2c.conv)
        # print(type(module.backbone.res_layers._modules['1'].blocks._modules['3'].branch2c.conv))
        
        # s3 = module.backbone.res_layers._modules['1'].blocks._modules['3'].branch2c.conv
        # s4 = module.backbone.res_layers._modules['2'].blocks._modules['5'].branch2c.conv
        # s5 = module.backbone.res_layers._modules['3'].blocks._modules['2'].branch2c.conv
        # target_layers = [s3, s4, s5]
        
        # for i, layer in enumerate(target_layers):
        #     cam = EigenCAM(module, [layer])
        #     grayscale_cam = cam(tensor)[0, :, :]
        #     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        #     Image.fromarray(cam_image)
        #     cv2.imwrite(f"cam_s{i+3}_conv.jpg", cam_image)
        
        
        return

'''
    module.backbone.res_layers._modules['1'].blocks._modules['3'].branch2c.act, \
    module.backbone.res_layers._modules['2'].blocks._modules['5'].branch2c.act, \
    module.backbone.res_layers._modules['3'].blocks._modules['2'].branch2c.act]
'''