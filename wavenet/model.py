"""
Networks and components that define WaveNet
"""
from collections import OrderedDict

import torch
import torch.nn as nn


class WaveNet(torch.nn.Module):
    """Full WaveNet implementation.

    Major architecture is based off of Figure 4 in:
    https://arxiv.org/abs/1609.03499
    """

    def __init__(self, in_channels: int, res_channels: int):
        """Initialize WaveNet.

        Args:
            in_channels: number of channels for input channel. The number of
                skip channels is the same as input channels.
            res_channels: number of channels for residual input and output
        """
        super().__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("causal_conv", CausalConv(in_channels, res_channels)),
                    (
                        "residual_stack",
                        ResidualStack(
                            in_channels=res_channels, out_channels=in_channels
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


class CausalConv(torch.nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, in_channels, out_channels):
        """Initialize causal convolution"""
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=2, stride=1, padding=1, bias=False
        )

    def forward(self, data):
        """Apply causal convolution"""
        return self.conv(data)


class ResidualStack(torch.nn.Module):
    """Stack of ResidualLayers"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dilation_loops: int = 3,
        dilation_limit: int = 9,
    ):
        """Initialize residual stack.

        Args:
            in_channels: size of inputs for residual stack
            out_channels: size of outputs for skip connections
            num_dilation_loops: Number of times that full list of dilation
                values will be repeated.
                e.g. [1, 2, ..., 2**n, 1, 2, ..., 2**n, ...]
            dilation_limit: Dilation values will be 2**n where n is an integer
                that climbs up to this limit. Default: 9
                e.g. [1, 2, 4, ..., 2**9]

        """
        super().__init__()
        self.num_dilation_loops = num_dilation_loops
        self.dilation_limit = dilation_limit
        self.residual_layers = self._create_residual_stack(in_channels, out_channels)

    def forward(self, data):
        """Iterate through residual layers stack"""
        output = data
        skip_connections = []

        for layer in self.residual_layers:
            # output is used for input to next residual layer
            # skip_output is used for actual evaluation from entire stack
            output, skip_output = layer(output)
            skip_connections.append(skip_output)

        skip_output = torch.sum(torch.stack(skip_connections), axis=0)

        return skip_output

    def _create_residual_stack(self, in_channels, out_channels):
        """Create list of ResidualLayers that will make up the stack.

        Args:
            in_channels: size of inputs for residual stack
            out_channels: size of outputs for skip connections

        Stack contains `self.num_dilation_loops * self.dilation_limit` number
        of residual layers. Where the dilations for each layer will have this
        format respectively:
        [1, 2, 4, ..., 2 ** self.dilation_limit,
         1, 2, 4, ..., 2 ** self.dilation_limit,
         ...
         (for self.num_dilation_loops times)]

        """
        stack = []
        for _ in range(self.num_dilation_loops):
            for dilation in range(self.dilation_limit):
                stack.append(
                    ResidualLayer(in_channels, out_channels, dilation=dilation)
                )

        return stack


class ResidualLayer(torch.nn.Module):
    """Single residual layer"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """Initialize a residual layer"""
        super().__init__()
        self.dilated_causal_conv = DilatedCausalConvolution(
            in_channels, out_channels, dilation
        )
        self.gated_activation = GatedActivationUnit()
        self.conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, data):
        """Pipeline for a single residual layer"""
        filter_output, gate_output = self.dilated_causal_conv(data)
        activation_output = self.gated_activation(filter_output, gate_output)
        skip_output = self.conv(activation_output)
        output = data + skip_output

        return output, skip_output


class DilatedCausalConvolution(torch.nn.Module):
    """Dilated causal convolution layer"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        """Initialize a dilated convolution.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            dilation: dilation size

        """
        super().__init__()
        self.conv_filter = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.conv_gate = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            dilation=dilation,
            padding=0,
            bias=False,
        )

    def forward(self, data):
        """Apply dilated convolutions and create filter and gates ouputs"""
        filter_output = self.conv_filter(data)
        gate_output = self.conv_gate(data)

        return filter_output, gate_output


class GatedActivationUnit(torch.nn.Module):
    """Gated activation unit layer from PixelCNN"""

    def __init__(self):
        """Initialize a gated activation unit"""
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, filter_data, gate_data):
        """Apply gated activation unit to dilated convolutions.

        From:
            https://arxiv.org/abs/1606.05328
            z = tanh(W_{f, k} ∗ x) ⊙ σ(W_{g,k} ∗ x)

        Where:
            ∗ denotes a convolution operator
            ⊙denotes and element-wise multiplication operator
            σ(·) is a sigmoid function
            k is the layer index
            f and g denote filter and gate respectively
            W is learnable convolution filter

        """
        output = self.tanh(filter_data) * self.sigmoid(gate_data)

        return output
