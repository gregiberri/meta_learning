import torch.nn
from ml.modules.backbones.erfnet import ERFNet
from ml.modules.backbones.resnet import ResNet


class Backbone(torch.nn.Sequential):
    def __init__(self, backbone_config):
        super(Backbone, self).__init__()

        if backbone_config.name == 'ResNet':
            self.add_module('resnet', ResNet(backbone_config.params))
        elif backbone_config.name == 'ERFNet':
            self.add_module('ERFNet', ERFNet(backbone_config.params))
        elif backbone_config.name == 'efficientnet':
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong backbone name: {backbone_config.name}')
