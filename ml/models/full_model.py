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
        self.domain = 'target_train'

        self.model = get_model(config)
        self.fc_source = nn.Linear(512 * self.model.block.expansion, self.config.params.num_source_classes)
        self.fc_target = nn.Linear(512 * self.model.block.expansion, self.config.params.num_target_classes)

        self.loss = get_classification_loss(self.config.params.loss)

    def set_domain(self, mode):
        if 'source' in mode:
            self.domain = 'source'
        elif 'target' in mode:
            self.domain = 'target'
        elif 'multitask' in mode:
            self.domain = 'multitask'
        else:
            raise ValueError(f'Wrong mode: {mode}')

    def get_pred(self, outputs):
        if self.domain == 'multitask':
            return [torch.argmax(output, -1) for output in outputs]
        else:
            return torch.argmax(outputs, -1)

    def get_accuracy(self, preds, targets):
        if self.domain == 'multitask':
            return torch.mean(torch.stack([torch.mean((pred == target).float()) for pred, target in zip(preds, targets)]))
        else:
            return torch.mean((preds == targets).float())

    def get_loss(self, pred, targets):
        if self.training:
            if self.domain == 'multitask':
                source_pred, target_pred = pred
                source_target, target_target = targets

                source_loss = self.loss(source_pred, source_target)
                target_loss = self.loss(target_pred, target_target)
                full_loss = 1 / 2 * (self.config.params.source_weight * source_loss + \
                                     self.config.params.target_weight * target_loss)

                return full_loss
            else:
                return self.loss(pred, targets)
        else:
            return torch.as_tensor(0)

    def get_output(self, output):
        if self.domain == 'source':
            return self.fc_source(output)
        elif self.domain == 'target':
            return self.fc_target(output)
        elif self.domain == 'multitask':
            data_split = [int(element) for element in self.config.params.data_split]
            source_output, target_output = torch.split(output, data_split, dim=0)
            source_output = self.fc_source(source_output)
            target_output = self.fc_target(target_output)

            output = [source_output, target_output]

            return output
        else:
            raise ValueError(f'Wrong domain{self.domain}')

    def forward(self, image, target):
        model_output = self.model(image)
        output = self.get_output(model_output)

        output_dict = dict()
        output_dict['output'] = output
        output_dict['loss'] = self.get_loss(output, target)
        output_dict['pred'] = self.get_pred(output)
        output_dict['accuracy'] = self.get_accuracy(output_dict['pred'], target)

        return output_dict
