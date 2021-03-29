"""base model which contains the model, and the loss
"""

import torch
import torch.nn as nn

from ml.models import get_model
from ml.modules.losses import get_classification_loss


class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = get_model(config)

        self.loss = get_classification_loss(self.config.params.loss)

    def set_domain(self, mode):
        if 'source' in mode:
            self.model.domain = 'source'
        elif 'target' in mode:
            self.model.domain = 'target'
        else:
            raise ValueError(f'Wrong mode: {mode}')

    def get_pred(self, output):
        return torch.argmax(output, -1)

    def get_accuracy(self, preds, targets):
        return torch.mean((preds == targets).float())

    def get_loss(self, output, targets):
        return self.loss(output, targets)

    def forward(self, image, target):
        output = self.model(image)

        output_dict = {'output': output}
        if self.training:
            output_dict['loss'] = self.get_loss(output, target)
        else:
            output_dict['loss'] = torch.as_tensor(0)

        output_dict['pred'] = self.get_pred(output)
        output_dict['accuracy'] = self.get_accuracy(output_dict['pred'], target)

        return output_dict


