import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

import torch.optim as optimizer_module


# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)


##########################
# Generic training class #
##########################
class Trainer(nn.Module):
    def __init__(self, log_loss_every=1000, writer=None):
        super(Trainer, self).__init__()
        self.iterations = 0

        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}

    def get_device(self):
        return list(self.parameters())[0].device

    def train_step(self, st_data, sub_data):
        # Set all the models in training mode
        self.train(True)

        # Log the values in loss_items every log_loss_every iterations
        if not (self.writer is None):
            if (self.iterations + 1) % self.log_loss_every == 0:
                self._log_loss()

        # Move the data to the appropriate device
        device = self.get_device()

        for i, item in enumerate(st_data):
            st_data[i] = item.to(device)

        for j in range(len(sub_data)):
            if sub_data[j] is not None:
                sub_data[j] = sub_data[j].to(device)

        # Perform the training step and update the iteration count
        self._train_step(st_data, sub_data)
        self.iterations += 1

    def _add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.add_scalar(tag=key, scalar_value=np.mean(values), global_step=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        items_to_save = self._get_items_to_store()
        items_to_save['iterations'] = self.iterations

        # Save the model and increment the checkpoint count
        torch.save(items_to_save, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path)
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

                # Move the optimizer parameters to the same correct device.
                # see https://github.com/pytorch/pytorch/issues/2830 for further details
                if isinstance(attribute, Optimizer):
                    device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
                    for state in attribute.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)

            # Otherwise just copy the value
            else:
                setattr(self, key, value)

    def _get_items_to_store(self):
        return dict()

    def _train_step(self, st_data, sub_data):
        raise NotImplemented()
