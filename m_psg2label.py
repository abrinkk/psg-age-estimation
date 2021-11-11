import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from base.base_model import BaseModel
from config import Config

class M_PSG2FEAT(BaseModel):
    def __init__(self, config):
        """A model to process epochs of polysomnography data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        # Attributes
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.return_pdf_shape = config.return_pdf_shape
        
        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
                nn.Conv2d(1, 32, (self.n_channels, 1), bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True))
        # self.channel_mixer = nn.Sequential(
        #         nn.Conv2d(self.n_channels, 32, (1, 1), bias = False),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU6(inplace=True))
        self.MobileNetV2 = MobileNetV2(num_classes = self.n_class)
        self.LSTM = nn.LSTM(128, 128, num_layers = 1, bidirectional = True)
        self.add_attention = AdditiveAttention(256, 512)
        self.linear_l = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p = config.do_f)
        # Label specific layers
        self.classify_bias_init = [50.0, 10.0]
        self.classify_l = nn.Linear(256, self.n_class * 2)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)
        
    def forward(self, X):
        """Forward call of model

        Args:
            X (Tensor): Input polysomnography epoch of size [batch_size, n_channels, 38400]

        Returns:
            dict: A dict {'pred': age predictions, 
                          'feat': latent space representation,
                          'alpha': additive attention weights,
                          'pdf_shape': shape of predicted age distribution (not used)}
        """
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        # Modified MobileNetV2
        X = self.MobileNetV2(X)
        # X.size() = [Batch_size, Feature_maps = 320, Channels = 1, Time = 5*60*16]
        # LSTM layer
        X = X.view(-1, X.size(1), 1, int(X.size(3) / (5*4)), 5*4)
        X = torch.squeeze(X.mean([4]), 2)
        X = X.permute(2, 0, 1)
        self.LSTM.flatten_parameters()
        X, _ = self.LSTM(X)
        # Attention layer
        X = X.permute(1, 0, 2)
        # Averaged features
        X_avg = torch.mean(X, 1)
        X_a, alpha = self.add_attention(X)
        # Linear Transform
        X_a = self.linear_l(F.relu(X_a))
        # Dropout
        X_a = self.dropout(X_a)
        # Linear
        C = self.classify_l(X_a)
        C = torch.squeeze(C, 1)
        if self.return_only_pred:
            return torch.unsqueeze(C[:, 0], 1)
        else:
            return {'pred': C[:, 0], 'feat': torch.cat((X_a, X_avg), 1), 'alpha': alpha, 'pdf_shape': C[:, 1]}

class M_PSG2FEAT_wn(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.classify_bias_init = [36.7500]
        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
                nn.Conv2d(1, 32, (self.n_channels, 1), bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True))
        self.wavenet = nn.Sequential(dilated_conv_block(32,64,(1,9),(1,1),(1,1),1,(1,2),config.do_f), 
                                     dilated_conv_block(64,128,(1,9),(1,1),(1,2),1,(1,2),config.do_f), 
                                     dilated_conv_block(128,128,(1,9),(1,1),(1,4),1,(1,2),config.do_f), 
                                     dilated_conv_block(128,256,(1,9),(1,1),(1,8),1,(1,2),config.do_f), 
                                     dilated_conv_block(256,256,(1,9),(1,1),(1,16),1,(1,2),config.do_f)) 
            
        self.dropout = nn.Dropout(p = config.do_f)
        self.classify_l = nn.Linear(256, self.n_class)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)

    def forward(self, X):
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        X = self.wavenet(X)
        # X.size() = [Batch_size, Feature_maps = 256, Channels = 1, Time = 5*60*16]
        X = X.mean([2, 3])
        X = self.dropout(X)
        C = self.classify_l(F.relu(X))
        C = torch.squeeze(C, 1)
        alpha = torch.ones(C.size(0), 5, 1) / 5.0
        if self.return_only_pred:
            return C
        else:
            return {'pred': C, 'feat': X, 'alpha': alpha}

class M_PSG2FEAT_simple(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.pre_n_class
        self.n_label = len(config.pre_label)
        self.return_only_pred = config.return_only_pred
        self.classify_bias_init = [36.7500]
        
        ### LAYERS ###
        self.channel_mixer = nn.Sequential(
                nn.Conv2d(1, 32, (self.n_channels, 1), bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True))
        self.CNN = nn.Sequential(ConvBNReLU(32, 64, (1,3), (1,2)),
                                 ConvBNReLU(64, 64, (1,3), (1,2)),
                                 ConvBNReLU(64, 128, (1,3), (1,2)),
                                 ConvBNReLU(128, 256, (1,3), (1,2)),
                                 ConvBNReLU(256, 256, (1,3), (1,1))
                                 )
        self.classify_l = nn.Linear(256, self.n_class)
        self.dropout = nn.Dropout(p = config.do_f)
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)
        
    def forward(self, X):
        # X.size() = [Batch_size, Channels = 13, Time = 5*60*128]
        X = torch.unsqueeze(X, 1)
        # X.size() = [Batch_size, Feature_maps = 1, Channels = 13, Time = 5*60*128]
        # Channel Mixer
        X = self.channel_mixer(X)
        # X.size() = [Batch_size, Feature_maps = 32, Channels = 1, Time = 5*60*128]
        X = self.CNN(X)
        # X.size() = [Batch_size, Feature_maps = 256, Channels = 1, Time = 5*60*16]
        X = X.mean([2, 3])
        # dropout
        X = self.dropout(X)
        # Classify layer
        C = self.classify_l(F.relu(X))
        C = torch.squeeze(C, 1)
        alpha = torch.ones(C.size(0), 5, 1) / 5.0
        if self.return_only_pred:
            return C
        else:
            return {'pred': C, 'feat': X, 'alpha': alpha}

class M_FEAT2LABEL(BaseModel):
    def __init__(self, config):
        """A model to process latent space representations of polysomnography data

        Args:
            config: An instance of the config class with set attributes.
        """
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.label_cond = sum(config.label_cond_size)
        self.return_only_pred = config.return_only_pred
        self.return_pdf_shape = config.return_pdf_shape
        
        ### LAYERS ###
        self.LSTM = nn.LSTM(512, 32 * config.net_size_scale, num_layers = config.lstm_n, bidirectional = True)
        self.add_attention = AdditiveAttention(64 * config.net_size_scale, 128 * config.net_size_scale)
        self.linear_l = nn.Linear(64 * config.net_size_scale + self.label_cond, 64 * config.net_size_scale)
        self.dropout = nn.Dropout(p = config.do_l)
        self.classify_l = nn.Linear(64 * config.net_size_scale, self.n_class * 2)
        self.classify_bias_init = [50.0, 10.0]
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)
#        self.transformer = nn.TransformerEncoderLayer(256, nhead = 8)
        
    def forward(self, X, z):
        """Forward call of model

        Args:
            X (Tensor): latent space representation of size [batch_size, n_epochs, n_features]
            z (Tensor): additional input scalars of size [batch_size, n_z]

        Returns:
            dict: A dict {'pred': age predictions, 
                          'alpha': additive attention weights,
                          'pdf_shape': shape of predicted age distribution (not used)}
        """
        # z.size() = [batch_size, label_cond]
        # X.size() = [batch_size, n_epochs, Features]
        X = X.permute(1, 0, 2)
        self.LSTM.flatten_parameters()
        X, _ = self.LSTM(X)
        # Attention layer
        X = X.permute(1, 0, 2)
        X, alpha = self.add_attention(X)
        # Concatenate with conditional labels
        X = torch.cat((X, z), 1)
        # Linear Transform
        X = self.linear_l(F.relu(X))
        # Dropout
        X = self.dropout(X)
        # Classify layer
        X = self.classify_l(F.relu(X))
        X = torch.squeeze(X, 1)
        if self.return_only_pred:
            return X[:,0]
        else:
            return {'pred': X[:, 0], 'alpha': alpha, 'pdf_shape': X[:, 1]}

class M_FEAT2LABEL_simple(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.n_class = config.n_class
        self.label_cond = sum(config.label_cond_size)
        self.return_only_pred = config.return_only_pred
        
        ### LAYERS ###
        self.linear_l = nn.Linear(512 + self.label_cond, 64 * config.net_size_scale)
        self.dropout = nn.Dropout(p = config.do_l)
        self.classify_l = nn.Linear(64 * config.net_size_scale, self.n_class)
        self.classify_bias_init = [36.7500]
        self.classify_l.bias.data = torch.Tensor(self.classify_bias_init)
#        self.transformer = nn.TransformerEncoderLayer(256, nhead = 8)
        
    def forward(self, X, z):
        # z.size() = [batch_size, label_cond]
        # X.size() = [batch_size, n_epochs, Features]
        X = X.mean(1)
        X = torch.cat((X, z), 1)
        alpha = torch.ones(X.size(0), 120, 1) / 5.0
        # Linear Transform
        X = self.linear_l(F.relu(X))
        # Dropout
        X = self.dropout(X)
        # Classify layer
        X = self.classify_l(F.relu(X))
        X = torch.squeeze(X, 1)
        if self.return_only_pred:
            return X
        else:
            return {'pred': X, 'alpha': alpha}

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, (1,2)],
                [6, 32, 2, (1,2)],
                [6, 64, 2, (1,2)],
                [6, 128, 1, (1,1)],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(input_channel, input_channel, stride=(1,2))]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else (1,1)
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
#        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=(1,1)))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
#        self.classifier = nn.Sequential(
#            nn.Dropout(0.2),
#            nn.Linear(self.last_channel, num_classes),
#        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
#        x = x.mean([2, 3])
#        x = self.classifier(x)
        return x

# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=(1,3), stride=(1,1), groups=1):
        padding = tuple((np.array(kernel_size) - 1) // 2)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

# Dilated conv block (conv->bn->relu->maxpool->dropout)
class dilated_conv_block(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), stride=(1,1), dilation=(1,1), groups=1, max_pool=(1,1), drop_chance=0.0):
        padding = tuple(((np.array(kernel_size)-1)*(np.array(dilation)-1) + np.array(kernel_size) - np.array(stride)) // 2)
        if max_pool[1] > 1:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation , groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=max_pool, stride=max_pool),
                nn.Dropout(p=drop_chance)
            )
        else:
            super(dilated_conv_block, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation , groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_chance)
            )

# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        """Additive attention module

        Args:
            input_size (int): Size of input
            hidden_size (int): Size of hidden layer
        """
        super(AdditiveAttention, self).__init__()
        self.linear_u = nn.Linear(input_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, h):
        """Forward call of model

        Args:
            h (Tensor): Input features of size [batch_size, ..., input_size]

        Returns:
            s (Tensor): Summary features
            a (Tensor): Additive attention weights
        """
        # h.size() = [Batch size, Sequence length, Hidden size]
        u = torch.tanh(self.linear_u(h))
        a = F.softmax(self.linear_a(u), dim = 1)
        s = torch.sum(a * h, 1)
        return s, a
        
# Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride[1] in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride[1] == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=(1,1)))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, (1,1), (1,1), 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == "__main__":
    # Set up config
    config = Config()
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    model_F = M_PSG2FEAT(config).to(device)
    model_L = M_FEAT2LABEL(config).to(device)

    # Model Debug
    model_F.debug_model((4, 12, 128*5*60), device)
    model_L.debug_model((64, 120, 128*2), device, cond_size = sum(config.label_cond_size))