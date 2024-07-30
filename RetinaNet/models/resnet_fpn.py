import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    "ResNet50_ADN_FPN",
    "ResNet101_ADN_FPN",
    
]


# Bottleneck block with stage-level shortcut connections
class BottleneckBlock_Skippable(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, \
        use_skip_bn=True, \
        norm_layer = None):
        super(BottleneckBlock_Skippable, self).__init__()
        
        self.use_skip_bn = use_skip_bn

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
      
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)

        self.bn1 = norm_layer(planes)
        self.bn2 = norm_layer(planes)
        self.bn3 = norm_layer(self.expansion*planes)

        # skip-aware batch normaliation
        if (self.use_skip_bn == True):
            self.bn1_skip = norm_layer(planes)
            self.bn2_skip = norm_layer(planes)
            self.bn3_skip = norm_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x, skip=False):
        if (skip == True):
            out = F.relu(self.bn1_skip(self.conv1(x)))
            out = F.relu(self.bn2_skip(self.conv2(out)))
            out = self.bn3_skip(self.conv3(out))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class SkippableSequential(nn.Sequential):
    def forward(self, input, skip=False):
        for i in range(len(self)):
            if ((skip == True) and (self[i].use_skip_bn == False)):
                pass
            else:
                input = self[i](input, skip=skip) 
 
        return input


class ResNet_Bottleneck_Skip(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, norm_layer=None):
        super(ResNet_Bottleneck_Skip, self).__init__()
        self.inplanes = 64

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_skippable = self._make_layer_with_branch(block, 64, num_blocks[0], stride=1)
        self.layer2_skippable = self._make_layer_with_branch(block, 128, num_blocks[1], stride=2)
        self.layer3_skippable = self._make_layer_with_branch(block, 256, num_blocks[2], stride=2)
        self.layer4_skippable = self._make_layer_with_branch(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (norm_layer, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_with_branch(self, block, planes, blocks, stride=1):
        layers = []

        norm_layer = self._norm_layer

        # the number of blocks are shared by super-net and sub-net
        n_shared = (blocks + 1) // 2 
        n_shared = min(4, n_shared)

        layers.append(block(self.inplanes, planes, stride, use_skip_bn=True, norm_layer=norm_layer))    
        self.inplanes = planes * block.expansion
        for b in range(1, blocks):
            _use_skip_bn = True
            if (b >= n_shared):
                _use_skip_bn = False
            layers.append(block(self.inplanes, planes, use_skip_bn=_use_skip_bn, norm_layer=norm_layer))

        return SkippableSequential(*layers)
 
    def forward(self, x, skip=(False,False,False,False)):
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.relu(out)

        out = self.maxpool(out) 

        out = self.layer1_skippable(out, skip=skip[0])
        out = self.layer2_skippable(out, skip=skip[1])
        out = self.layer3_skippable(out, skip=skip[2])
        out = self.layer4_skippable(out, skip=skip[3])
        
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
     
        return out


def ResNet50_ADN_FPN(norm_layer=None):
    return ResNet_Bottleneck_Skip(BottleneckBlock_Skippable, [3, 4, 6, 3], norm_layer=norm_layer)
def ResNet101_ADN_FPN(norm_layer=None):
    return ResNet_Bottleneck_Skip(BottleneckBlock_Skippable, [3, 4, 23, 3], norm_layer=norm_layer)


def test():
    net = ResNet50_ADN_FPN()
    x = torch.randn(16,3,224,224) 

    print(net)
    #y = net(x, skip=(True,True,True,True))  # base-net
    y = net(x, skip=(False,False,False,False))  # super-net
    print(y.size())


if __name__ == '__main__':
    test()
