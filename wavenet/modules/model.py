"""
Networks and components that define WaveNet
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import wavenet.utils as utils
from wavenet.modules import CausalConv, ResidualStack


class WaveNet(torch.nn.Module):
    """Full WaveNet implementation.

    Major architecture is based off of Figure 4 in:
    https://arxiv.org/abs/1609.03499
    """

    def __init__(self, in_channels: int, res_channels: int, log_level: int = 20):
        """Initialize WaveNet.

        Args:
            in_channels: number of channels for input channel. The number of
                skip channels is the same as input channels.
            res_channels: number of channels for residual input and output
            log_level: logging level

        """
        super().__init__()
        self.logger = utils.new_logger(self.__class__.__name__, level=log_level)
        self.causal_conv = CausalConv(in_channels, res_channels, kernel_size=2)
        self.residual_stack = ResidualStack(
            res_channels=res_channels, skip_channels=in_channels
        )
        self.conv_1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu_2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through full architecture"""
        data = self.causal_conv(data)
        data = self.residual_stack(data)
        data = self.conv_1(data)
        data = self.relu_1(data)
        data = self.conv_2(data)
        data = self.relu_2(data)
        data = self.softmax(data)

        return data
