"""
Utilities for neural networks.
"""
import torch.nn as nn
from torchvision import models

from .custom_nets import ResNet, ResNetBasicBlock
from .common_utils import PrintStyle

'''
This function should create a XResNet (currently a normal ResNet)

Source for full resnet architecture: https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
Source for feature extractor: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''


def create_feature_extractor(n_features: int, base_net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)):
    """
    Adapts a (potentially pre-trained) ResNet into a given number of features with a Linear Module.

    Parameters
    ----------
    n_features :
        Output size of the reconverted model.
    base_net :
        Original model.

    Returns
    -------
    out:
        Feature extractor for transfer learning.
    """
    for param in base_net.parameters():
        param.requires_grad = False

    num_ftrs = base_net.fc.in_features
    base_net.fc = nn.Linear(num_ftrs, n_features)

    return base_net


def create_full_net(input_channels: int = None, output_length: int = None,
                    block=ResNetBasicBlock, depths=None, pytorch_net: bool = True, *args, **kwargs):
    """
    Creates the ResNet architecture. ResNet-18 by default

    Parameters
    ----------
    input_channels : int
        Channels of the image (Number of detectors).
    output_length : int
        If alone: Number of parameters to do the regression on.
        If hooked to normalizing flow: Number of features to train the flow on.
    block : object, optional
        Block type for the net. The default is ResNetBasicBlock.
    depths : list[int], optional
        Number of blocks in each layer. The default is [2, 2, 2, 2].
    pytorch_net :
        Whether to use pytorch's ResNet class.

    Returns
    -------
    object
        The residual network.

    """

    if depths is None:
        depths = [2, 2, 2, 2]
    if pytorch_net:
        if block not in [models.resnet.BasicBlock, models.resnet.Bottleneck]:
            block = models.resnet.BasicBlock
        try:
            output_features = kwargs.pop('output_features')
        except KeyError as exc:
            output_features = 1000
            print(PrintStyle.red + 'WARNING: You need to provide "output_features" to build a flow from a PyTorch '
                                   f'ResNet\nSetting argument to default ({output_features})' + PrintStyle.reset)
        net = models.ResNet(block=block, layers=depths, *args, **kwargs)
        net.fc = nn.Linear(in_features=512 * block.expansion, out_features=output_features)
        return net

    if (input_channels is None) or (output_length is None):
        raise ValueError('Non-PyTorch ResNets need arguments "input_channels" and "output_length".')

    return ResNet(input_channels, output_length, block=block, deepths=depths, *args, **kwargs)



