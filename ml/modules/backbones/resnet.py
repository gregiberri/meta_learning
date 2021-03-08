import torch
from torchvision.models import resnet



class ResNet(torch.nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        pretrained_model = resnet.__dict__['resnet{}'.format(34)](pretrained=config['pretrained'])
        self.conv1 = conv_bn_relu(3,
                                  64,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        self.conv6 = conv_bn_relu(512,
                                  512,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        del pretrained_model  # clear memory

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        feat = self.conv_final(y)

        return feat