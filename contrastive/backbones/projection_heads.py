from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn


class ProjectionHead(pl.LightningModule):

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256,10],
                 activation='linear',
                 drop_rate=0):
        super(ProjectionHead, self).__init__()
        self.num_representation_features = num_representation_features # useless ?

        # define layers
        layers = []
        input_size = layers_shapes[0]

        for i, dim_i in enumerate(layers_shapes[1:]):
            output_size = dim_i
            layers.append(
                ('Linear%s' % i, nn.Linear(input_size, output_size)))
            
            # add activation after each layer except last
            if i == len(layers_shapes)-2:
                pass
            else:
                #layers.append((f'BatchNorm{i}', nn.BatchNorm1d(output_size)))
                if activation == 'linear':
                    pass
                elif activation == 'relu':
                    layers.append((f'LeakyReLU{i}', nn.LeakyReLU()))
                elif activation == 'sigmoid':
                    layers.append((f'Sigmoid{i}', nn.Sigmoid()))
                else:
                    raise ValueError(f"The given activation '{activation}' is not \
    handled. Choose between 'linear', 'relu' or 'sigmoid'.")
                layers.append((f'DropOut{i}', nn.Dropout(p=drop_rate)))
            
            input_size = output_size
        
        self.layers = nn.Sequential(OrderedDict(layers))

        # for m in self.layers:
        #     if isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.5)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return out