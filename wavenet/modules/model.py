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
        self.logger = utils.new_logger("WaveNet Model", level=log_level)
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("causal_conv", CausalConv(in_channels, res_channels)),
                    (
                        "residual_stack",
                        ResidualStack(
                            res_channels=res_channels, skip_channels=in_channels
                        ),
                    ),
                    ("conv_1", nn.Conv1d(in_channels, in_channels, kernel_size=1)),
                    ("relu_1", nn.ReLU()),
                    ("conv_2", nn.Conv1d(in_channels, in_channels, kernel_size=1)),
                    ("relu_2", nn.ReLU()),
                    ("softmax", nn.Softmax()),
                ]
            )
        )

    def forward(self, data):
        """Forward pass through full architecture"""
        return self.model(data)
