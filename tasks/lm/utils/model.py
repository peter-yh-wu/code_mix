"""
https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
"""

import torch.nn as nn
import torch.nn.init as init


def weight_init(module):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(module, nn.Conv1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.BatchNorm1d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm3d):
        init.normal_(module.weight.data, mean=1, std=0.02)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight.data)
        init.normal_(module.bias.data)
    elif isinstance(module, nn.LSTM):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.LSTMCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRU):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRUCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)