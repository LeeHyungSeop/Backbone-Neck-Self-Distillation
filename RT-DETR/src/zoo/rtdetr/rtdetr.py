"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]    

@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None, wNeck=False):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        '''
        for i in range(len(x)):
            print(f"\tx[{i}] : {x[i].shape}")
                
            x[0] : torch.Size([4,  512, 4h, 4w])
            x[1] : torch.Size([4, 1024, 2h, 2w])
            x[2] : torch.Size([4, 2048,  h,  w])
        '''   
        
        backbone_outs, neck_outs = self.encoder(x, wNeck=wNeck)
        '''
        for i in range(len(x)):
            print(f"\tx[{i}] : {x[i].shape}")
            
            x[0] : torch.Size([4, 256, 4h, 4w])
            x[1] : torch.Size([4, 256, 2h, 2w])
            x[2] : torch.Size([4, 256,  h,  w])
            
            notation of hybrid_encoder.py :
                x = outs = [fusion_F5-S4-S3, fusion_F5-S4-S3-S4, fusion_F5-S4-S3-S4-F5]
                    fusion_F5-S4-S3.shape       = [b, 256, 4h, 4w]
                    fusion_F5-S4-S3-S4.shape    = [b, 256, 2h, 2w]
                    fusion_F5-S4-S3-S4-F5.shape = [b, 256,  h,  w]
        ''' 
        
        if wNeck :
            x = neck_outs
            # # 2024.08.05 @hslee : skip connection 
            # x[0] = x[0] + backbone_outs[0]
            # x[1] = x[1] + backbone_outs[1]
            # x[2] = x[2] + backbone_outs[2]
        else : 
            x = backbone_outs
            
        x = self.decoder(x, targets)
        '''
        # print(f"[Final Output]")
        for key, value in x.items():
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
        '''
        
        # 2024.07.25 @hslee : add backbone_outs, neck_outs to x(dict_keys)
        if wNeck :
            x.update({'backbone_outs': backbone_outs, 'neck_outs': neck_outs})

        
        return x['pred_boxes']
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
