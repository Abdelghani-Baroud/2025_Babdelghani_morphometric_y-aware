from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import numpy as np

def ComputeOutputDim(dimension, depth):
    """Compute the output resolution
    """
    if depth==0:
        return dimension
    else:
        return(ComputeOutputDim(dimension//2+dimension%2, depth-1))

class Conv3dSame(nn.Conv3d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((np.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        ih, iw, id = x.size()[-3:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=id, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(
                x, [int(pad_d // 2), int(pad_d - pad_d // 2),
                    int(pad_w // 2), int(pad_w - pad_w // 2),
                    int(pad_h // 2), int(pad_h - pad_h // 2)]
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout3d_always(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Alwyas applies dropout also during evaluation

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
                  (same shape as input).

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, True, self.inplace)


class ConvNet(pl.LightningModule):
    r"""3D-ConvNet model class, based on

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels=1, encoder_depth=3, block_depth=2,
                 num_representation_features=256, linear=True,
                 adaptive_pooling=None, filters=[16,32,64], initial_kernel_size=3,
                 initial_stride=1, max_pool=False, drop_rate=0.1, memory_efficient=False,
                 in_shape=None):

        super(ConvNet, self).__init__()

        self.num_representation_features = num_representation_features
        self.drop_rate = drop_rate

        # Decoder part
        self.in_shape = in_shape
        c, h, w, d = in_shape
        self.encoder_depth = encoder_depth
        self.filters = filters
        self.block_depth = block_depth
        self.initial_kernel_size = initial_kernel_size
        self.initial_stride = initial_stride
        self.max_pool = max_pool
        assert len(self.filters) >= encoder_depth, "Incomplete filters list given."

        if adaptive_pooling is None:
            # receptive field downsampled
            #self.z_dim_h = h//2**self.encoder_depth
            #self.z_dim_w = w//2**self.encoder_depth
            #self.z_dim_d = d//2**self.encoder_depth
            self.z_dim_h = ComputeOutputDim(h, self.encoder_depth)
            self.z_dim_w = ComputeOutputDim(w, self.encoder_depth)
            self.z_dim_d = ComputeOutputDim(d, self.encoder_depth)
            self.out_dim = self.z_dim_h*self.z_dim_w*self.z_dim_d
        else:
            self.out_dim = np.prod(adaptive_pooling[1])

        modules_encoder = []
        layer_name = ['', 'a', 'b', 'c']
        for step in range(encoder_depth):
            for depth in range(block_depth-1):
                name = layer_name[depth]
                in_channels = 1 if (step == 0 and depth==0) else out_channels
                kernel_size = self.initial_kernel_size if (step == 0 and depth==0) else 3
                stride = self.initial_stride if (step==0 and depth==0) else 1
                out_channels = filters[step]
                #out_channels = 16 if step == 0 else 16 * (2**step)
                modules_encoder.append(
                    (f'conv{step}{name}',
                    nn.Conv3d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
                    ))
                modules_encoder.append(
                    (f'norm{step}{name}', nn.BatchNorm3d(out_channels)))
                modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
                if (self.max_pool and step == 0 and depth==0):
                    modules_encoder.append(('MaxPool', nn.MaxPool3d((2,2,2))))
                modules_encoder.append(
                    (f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))

            name=layer_name[block_depth-1]
            modules_encoder.append(
                (f'conv{step}{name}',
                Conv3dSame(in_channels=out_channels, out_channels=out_channels,
                        kernel_size=(3,3,3), stride=(2,2,2), groups=1, bias=True)
                ))
            modules_encoder.append(
                (f'norm{step}{name}', nn.BatchNorm3d(out_channels)))
            modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
            modules_encoder.append(
                (f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels
        # adaptive pool to ensure a fixed size linear layer accross regions
        if adaptive_pooling is not None:
            if adaptive_pooling[0]=='max':
                    modules_encoder.append(('AdaptiveMaxPool', nn.AdaptiveMaxPool3d(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0]=='average':
                    modules_encoder.append(('AdaptiveAvgPool', nn.AdaptiveAvgPool3d(output_size=adaptive_pooling[1])))
            else:
                raise ValueError("Wrong pooling name argument")
        # flatten and reduce to the desired dimension
        modules_encoder.append(('Flatten', nn.Flatten()))
        if linear:
            modules_encoder.append(
                ('Linear',
                nn.Linear(
                    self.num_features*self.out_dim,
                    self.num_representation_features)
                ))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

    def forward(self, x):
        out = self.encoder(x)
        return out.squeeze(dim=1)
