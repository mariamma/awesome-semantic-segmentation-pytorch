"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from ..nn import JPU
from .base_models.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
from .base_models.resnet import resnet50, resnet101

import torch

__all__ = ['SegBaseModel']

class ObjectDetection(nn.Module):
    def __init__(self):
        super(ObjectDetection,self).__init__()
        self.transition_layer = nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False)
        self.gap = nn.AvgPool2d(16, 16)
        # self.prediction_layer = nn.Sequential(nn.Linear(2048, 15), nn.Sigmoid())
        self.prediction_layer = nn.Sequential(nn.Linear(2048, 15))
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,x):
        x = self.transition_layer(x)
        x = self.gap(x)
        x = torch.squeeze(x)
        out1 = self.prediction_layer(x)
        out2 = self.sigmoid_layer(out1)
        return out1, out2

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        self.objectDetection = ObjectDetection()
        if backbone == 'resnet50':
            
            if 'local_rank' in kwargs:
                kwargs.pop('local_rank', None)
            # self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.pretrained =  resnet50(pretrained=pretrained_base, **kwargs) 
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
