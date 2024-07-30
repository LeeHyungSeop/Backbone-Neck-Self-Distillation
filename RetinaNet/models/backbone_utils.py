import warnings
from typing import Callable, Dict, List, Optional, Union

from torch import nn, Tensor
from .misc import FrozenBatchNorm2d
from .feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

from .resnet import resnet50, resnet101
from ._api import _get_enum_from_fn, WeightsEnum
from ._utils import handle_legacy_interface, IntermediateLayerGetter

# 2024.03.21 @hslee
class BackboneWithADNFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels
        
        # 2024.03.22 @hslee
        # print(f"self.body : {self.body}") # IntermediateLayerGetter()
        # print(f"self.fpn : {self.fpn}") # FeaturePyramidNetwork()
        # print(f"self.out_channels : {self.out_channels}") # 256

    # 2024.03.22 @hslee
    def forward(self, x: Tensor, skip=None) -> Dict[str, Tensor]:
        # print(f"skip: {skip}") 
        # print(f"x.shape: {x.shape}")  # (bs, 3, 800, 1216)
        
        # 1. IntermediateLayerGetter() -> backbone's body
        x = self.body(x, skip=skip)  # self.body() returns OrderedDict type. 'model_out',  'features'
            # 'model_out' : resnet50's output tensor
            # 'features' : resnet50's feature maps for lateral connections in FPN
            
            # Trouble(mat1 and mat2 shapes cannot be multiplied (32768x1 and 2048x1000))
            # i deleted last FC layer in resnet50.py
            # because resnet50 for RetinaNet backbone is not needed FC layer(1000 classes classificayion) anymore. 
            
        # 2. FeaturePyramidNetwork() -> backbones's fpn
        x = self.fpn(x['features']) # x: Dict[str, Tensor]
        return x

@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: _get_enum_from_fn(resnet50.__dict__[kwargs["backbone_name"]])["IMAGENET1K_V1"],
    ),
)
def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: Optional[WeightsEnum],
    norm_layer: Callable[..., nn.Module] = FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithADNFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Args:
        backbone_name (string): resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        weights (WeightsEnum, optional): The pretrained weights for the model
        norm_layer (callable): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default, a ``LastLevelMaxPool`` is used.
    """
    backbone = resnet50.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    return _resnet50_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks)


def _resnet50_fpn_extractor(
    backbone: resnet50,
    trainable_layers: int,                          # 3
    returned_layers: Optional[List[int]] = None,    # [2, 3, 4]
    extra_blocks: Optional[ExtraFPNBlock] = None,   # LastLevelP6P7()
        # P6 is obtained via a 3×3 stride-2 conv on C5, 
        # P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6.
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> BackboneWithADNFPN:
    
    # print(f"extra_blocks: {extra_blocks}") 

    # making sure all `forward` function outputs participate in calculating loss. 
    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)    

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    
    # 2024.03.21 @hslee
    return BackboneWithADNFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )


def _validate_trainable_layers(
    is_trained: bool,
    trainable_backbone_layers: Optional[int],
    max_value: int,
    default_value: int,
) -> int:
    # don't freeze any layers if pretrained model or backbone is not used
    if not is_trained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has no effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                f"falling back to trainable_backbone_layers={max_value} so that all layers are trainable"
            )
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value # return 3 (default_value) @ 2024.03.21 hslee
    if trainable_backbone_layers < 0 or trainable_backbone_layers > max_value:
        raise ValueError(
            f"Trainable backbone layers should be in the range [0,{max_value}], got {trainable_backbone_layers} "
        )  
        
    print(f"trainable_backbone_layers : {trainable_backbone_layers}")
    return trainable_backbone_layers