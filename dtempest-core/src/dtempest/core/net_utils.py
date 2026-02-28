import torch.nn as nn

from torchvision import models

from .net_blocks import ResNetBasicBlock
from .common_utils import PrintStyle

'''
This function should create a XResNet (currently a normal ResNet)

Source for full resnet architecture: https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
Source for feature extractor: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''


def create_feature_extractor(n_features, base_net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)):
    for param in base_net.parameters():
        param.requires_grad = False

    num_ftrs = base_net.fc.in_features
    base_net.fc = nn.Linear(num_ftrs, n_features)

    return base_net


def create_full_net(input_channels: int = None, output_length: int = None,
                    block=ResNetBasicBlock, depths=[2, 2, 2, 2], pytorch_net: bool = True, *args, **kwargs):
    '''
    
    Creates the ResNet architecture. ResNet-18 by default

    Parameters
    ----------
    input_channels : int
        Channels of the image (Number of detectors).
    output_length : int
        If alone: Number of parameters to do the regresion on.
        If hooked to normalizing flow: Number of features to train the flow on
    block : object, optional
        Block type for the net. The default is ResNetBasicBlock.
    depths : list[int], optional
        Number of blocks in each layer. The default is [2, 2, 2, 2].
    pytorch_net :
        whether to use pytorch's ResNet class

    Returns
    -------
    object
        The resisual network.

    '''

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


class ResNet(nn.Module):
    '''
    The ResNet is composed of an encoder and a decoder
    '''

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by subsequent layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))  # Maybe-DO: substitute with blurpool
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
