import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .misc import MLP, Permute
from .stochastic_depth import StochasticDepth
from .utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
from ._utils import _ovewrite_named_param

from src.core import register

__all__ = [
    "SwinTransformer",
    "swin_t",
]


def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.norm_skip = norm_layer(4 * dim)  # wchkang

    def forward(self, x: Tensor, skip = False):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        if skip == True:        # wchkang
            x = self.norm_skip(x)
        else:
            x = self.norm(x)

        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
        skippable=False,  # @wooochul: is this block skippable? 
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.skippable = skippable 

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        
        if self.skippable == False:   # Skip-aware norm. for shared layers. @woochul
            self.norm1_skip = norm_layer(dim)
            self.norm2_skip = norm_layer(dim)  

    def forward(self, x: Tensor, skip = False):  
        if self.skippable == False and skip == True:
            # No stochastic depths for base-net. @woochul
            x = x + self.attn(self.norm1_skip(x))
            x = x + self.mlp(self.norm2_skip(x))
            # print(f"\t\t\t mandoatory path : {x.shape}")
        else:
            x = x + self.stochastic_depth(self.attn(self.norm1(x)))
            x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
            # print(f"\t\t\t skippable path : {x.shape}")

        return x



class SkippableSequentialBlocks(nn.Sequential):
    """Skips some blocks in the stage"""

    def forward(self, input, skip = False):
        """Extends nn.Sequential's forward for skipping some blocks
        Args:
            x (Tensor): input tensor
            skip (bool): if True, skip the last half blocks in the stage.
        """

        # print(f"\t len(self): {len(self)}")
        
        for idx, i in enumerate(range(len(self))):
            # print(f"\t\t idx: {idx}")
            if self[i].skippable == True and skip == True:
                # print(f"--skip {type(self[i])}")
                pass
            else:
                # print(f"--execute {type(self[i])}")
                input = self[i](input, skip)
        return input

class SkippableSequentialStages(nn.Sequential):
    """Pass 'skip' flag to selected stages"""

    def __init__(self, num_stages, num_patch_merginig, *args):
        super().__init__(*args)
        self.num_stages = num_stages 
        self.num_patch_merging = num_patch_merginig

    def forward(self, input, skip = None):
        """Extened forward to set "skip" flag for the selected stages.
        Args:
            input (Tensor): input tensor
            skip (List[bool]): list of stages to skip its half blocks
        """
        
        assert len(skip) == self.num_stages, \
            f"The networks has {self.num_stages} skippable stages, got: {len(skip)}"
        
        outs = []
        # print(f"len(self): {len(self)}") # 7
        for i in range(len(self)):
            # print(f"(i={i}) skip {skip[(i + 1) // 2]} {type(self[i])}")
            
            # self[i] : SkippableSequentialBlocks
            input = self[i](input, skip[(i + 1) // 2])
            if i != 0 and type(self[i]) == SkippableSequentialBlocks:
                outs.append(input)
            
        return outs

@register
class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        depths: List[int] = [2, 2, 6, 2],
        depths_base: List[int] = [1, 1, 3, 1],  
        num_heads: List[int]= [3, 6, 12, 24],
        patch_size: List[int] = [4, 4], 
        window_size: List[int] = [7, 7],
        embed_dim: int = 96,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        freeze_at=-1, 
        pretrained: bool = False):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        self.num_skippable_stages = len(depths)

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        self.pre =     nn.Sequential( 
            nn.Conv2d(
                3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0

        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                
                # @woochul: exp-sd-01: reverse order -> later blocks are more important in ADN
                # sd_prob = stochastic_depth_prob * float(total_stage_blocks - 1 - stage_block_id) / (total_stage_blocks - 1)

                # @woochul: exp-sd-02: per stage sd probability
                # sd_prob = stochastic_depth_prob * float(i_layer) / (depths[i_stage] - 1)

                # @woochul: exp-sd-03: fixed sd probability
                # sd_prob = stochastic_depth_prob
                # print(f"sd_prob:{sd_prob}")

                skippable = False

                if i_layer >= depths_base[i_stage]:
                    skippable = True

                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        skippable = skippable # wchkang
                    )
                )
                stage_block_id += 1
            layers.append(SkippableSequentialBlocks(*stage))
            # add patch merging lfayer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        
        # print(f"len(depths): {len(depths)}") 4
        # print(f"*layers: {len(layers)}")    [2, 2, 6, 2]
        self.features = SkippableSequentialStages(
            len(depths), len(depths) - 1, *layers)  

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        
        self.permute = Permute([0, 3, 1, 2])
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.flatten = nn.Flatten(1)
        # self.head = nn.Linear(num_features, num_classes)

        if pretrained:
            
            url = "https://download.pytorch.org/models/swin_t-704ceda3.pth"
            state = torch.hub.load_state_dict_from_url(url, progress=True)
            
            # new_state because of only the different key names
            new_state = {}
            for k, v in state.items():
                new_key = k
                
                if 'features' in k:
                    num = int(k.split('.')[1])
                    if num == 0:
                        new_key = k.replace(f'features.{num}', 'pre')
                    else:
                        new_key = k.replace(f'features.{num}', f'features.{num-1}')
                
                new_state[new_key] = v
                # print(f"key: {new_key}, value: {v.shape}")
                
            # 2024.06.16 @hslee
            del new_state["head.weight"]
            del new_state["head.bias"]
            
            # print all param
            # for k, v in new_state.items():
            #     print(f"key: {k}, value: {v.shape}")
                
            self.load_state_dict(new_state, strict=False) # strict=False because of only the absence of bn_skip
            print(f"Load state_dict from {url}")
        
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, len(depths))):
                self._freeze_parameters(self.res_layers[i])

        def _freeze_parameters(self, m: nn.Module):
            for p in m.parameters():
                p.requires_grad = False


    def forward(self, x, skip=None):
        
        outs = []

        # print(f"\t(raw input) {x.shape}")        
        
        x = self.pre(x) # stem
        # print(f"\t(after pre) {x.shape}")
        
        outs = self.features(x, skip=skip)
        for i in range(len(outs)):
            outs[i] = self.permute(outs[i]).contiguous()
            ''''
                    outs[0] torch.Size([4, 192, 80, 80])
                    outs[1] torch.Size([4, 384, 40, 40])
                    outs[2] torch.Size([4, 768, 20, 20])
            '''
            
        # x = self.norm(x)
        # # print(f"\t(after norm) {x.shape}")
        
        # x = self.permute(x)
        # # print(f"\t(after permute) {x.shape}")
        
        # x = self.avgpool(x)
        # # print(f"\t(after avgpool) {x.shape}")
        
        # x = self.flatten(x)
        # # print(f"\t(after flatten) {x.shape}")
        
        # x = self.head(x)
        # print(f"\t(after head) {x.shape}")
        
        '''
            (raw input) torch.Size([8, 3, 224, 224])
            (after pre) torch.Size([8, 56, 56, 96])
            (after features) torch.Size([8, 7, 7, 768])
            (after norm) torch.Size([8, 7, 7, 768])
            (after permute) torch.Size([8, 768, 7, 7])
            (after avgpool) torch.Size([8, 768, 1, 1])
            (after flatten) torch.Size([8, 768])
            (after head) torch.Size([8, 1000])
        '''
        
        return outs


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    depths_base: List[int],  # @wchkang: depths of base_net
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        depths_base=depths_base,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def swin_t(*, weights = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        depths_base=[1, 1, 3, 2], # @woochul: better choice!
        # depths_base=[1, 1, 4, 2], # @experiment #1: mandatory > skippable
        # depths_base=[1, 1, 2, 2], # @experiment #2: mandatory < skippable
        #depths_base=[1, 1, 3, 1], # @woochul: skpping in the last stage has much worse results.
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        #stochastic_depth_prob=0.2,  # pytorch's default value
        #stochastic_depth_prob=0.08,  
        # stochastic_depth_prob=0.09, 
        stochastic_depth_prob=0.1,  # @ wooochul  #very good!  
        #stochastic_depth_prob=0.12,  # @ wooochul  
        weights=weights,
        progress=progress,
        **kwargs,
    )

    