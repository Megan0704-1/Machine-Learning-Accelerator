import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                raise NotImplementedError
                # TODO: fill-in (end)

            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                raise NotImplementedError
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (end)
                raise NotImplementedError
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (end)
                raise NotImplementedError
                # TODO: fill-in (end)
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    raise NotImplementedError
    # TODO: fill-in (end)


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    
    # TODO: fill-in (start)
    raise NotImplementedError
    # TODO: fill-in (end)

