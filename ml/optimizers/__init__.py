from torch_optimizer import RAdam
from torch.optim import Adam, SGD, Adagrad, RMSprop

from ml.optimizers.lr_schedulers import StepLR, MultiStepLR, ConstantLR, PolyLR, WarmUpLR


def get_optimizer(optimizer_config, model_params):
    if optimizer_config.name == 'sgd':
        return SGD(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'adam':
        return Adam(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'radam':
        return RAdam(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'adagrad':
        return Adagrad(params=model_params, **optimizer_config.params.dict())
    elif optimizer_config.name == 'rmsprop':
        return RMSprop(params=model_params, **optimizer_config.params.dict())
    else:
        raise ValueError(f'Wrong optimizer name: {optimizer_config.name}')


def get_lr_policy(lr_policy_config, optimizer):
    if lr_policy_config.name == 'step':
        return StepLR(optimizer=optimizer, **lr_policy_config.params.dict())
    elif lr_policy_config.name == 'multistep':
        return MultiStepLR(optimizer=optimizer, **lr_policy_config.params.dict())
    elif lr_policy_config.name == 'constant':
        return ConstantLR(optimizer=optimizer)
    elif lr_policy_config.name == 'poly':
        return PolyLR(optimizer=optimizer, **lr_policy_config.params.dict())
    elif lr_policy_config.name == 'warmup':
        return WarmUpLR(optimizer=optimizer, **lr_policy_config.params.dict())
    else:
        raise ValueError(f'Wrong lr_policy name: {lr_policy_config.name}')
