import numpy as np
import torch
import vak

from .network import TweetyNet


def acc(y_pred, y):
    return y_pred.eq(y.view_as(y_pred)).sum().item() / np.prod(y.shape)


class TweetyNetModel(vak.Model):
    @classmethod
    def from_config(cls, config, logger=None):
        network = TweetyNet(**config['network'])
        loss = torch.nn.CrossEntropyLoss(**config['loss'])
        optimizer = torch.optim.SGD(params=network.parameters(), **config['optimizer'])
        metrics = {'acc': vak.metrics.Accuracy(),
                   'loss': torch.nn.CrossEntropyLoss()}
        return cls(network=network, optimizer=optimizer, loss=loss, metrics=metrics, logger=logger)
