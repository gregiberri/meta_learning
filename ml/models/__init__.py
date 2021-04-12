from ml.models import resnet, densenet


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: model_config namespace, containing the name and the params

    :return: model
    """

    if model_config.name == 'resnet18':
        return resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    elif model_config.name == 'resnet34':
        return resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3])
    elif model_config.name == 'resnet50':
        return resnet.ResNet(resnet.BottleNeck, [3, 4, 6, 3])
    elif model_config.name == 'resnet101':
        return resnet.ResNet(resnet.BottleNeck, [3, 4, 23, 3])

    elif model_config.name == 'densenet121':
        raise NotImplementedError()
        return densenet.DenseNet(densenet.Bottleneck, [6, 12, 24, 16], growth_rate=32)

    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')
