"""Convolution layers for WaveNet"""
import torch
import torch.nn as nn
import wavenet
from wavenet.modules import GatedActivationUnit


class CausalConv(torch.nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, in_channels, res_channels):
        """Initialize causal convolution.

        Args:
            in_channels: the number of channels for original input
            res_channels: the number of channels for the residual stack
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels, res_channels, kernel_size=2, stride=1, padding=1, bias=False
        )

    def forward(self, data):
        """Apply causal convolution"""
        return self.conv(data)


class ResidualStack(torch.nn.Module):
    """Stack of ResidualLayers."""

    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        num_dilation_loops: int = 3,
        dilation_limit: int = 9,
    ):
        """Initialize residual stack.

        Args:
            res_channels: number of channels for inputs to residual stack
            skip_channels: number of channels for skip connections
            num_dilation_loops: Number of times that full list of dilation
                values will be repeated.
                e.g. [1, 2, ..., 2**n, 1, 2, ..., 2**n, ...]
            dilation_limit: Dilation values will be 2**n where n is an integer
                that climbs up to this limit. Default: 9
                e.g. [1, 2, 4, ..., 2**9]

        """
        super().__init__()
        self.residual_layers = self._create_residual_stack(
            res_channels, skip_channels, num_dilation_loops, dilation_limit
        )

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

    @staticmethod
    def _create_residual_stack(
        res_channels: int,
        skip_channels: int,
        num_dilation_loops: int,
        dilation_limit: int,
    ):
        """Create list of ResidualLayers that will make up the stack.

        Args:
            res_channels: number of channels for inputs to residual stack
            skip_channels: number of channels for skip connections
            num_dilation_loops: Number of times that full list of dilation
                values will be repeated.
                e.g. [1, 2, ..., 2**n, 1, 2, ..., 2**n, ...]
            dilation_limit: Dilation values will be 2**n where n is an integer
                that climbs up to this limit. Default: 9
                e.g. [1, 2, 4, ..., 2**9]

        Stack contains `self.num_dilation_loops * self.dilation_limit` number
        of residual layers. Where the dilations for each layer will have this
        format respectively:
        [1, 2, 4, ..., 2 ** self.dilation_limit,
         1, 2, 4, ..., 2 ** self.dilation_limit,
         ...
         (for self.num_dilation_loops times)]

        """
        stack = []
        for _ in range(num_dilation_loops):
            for dilation in range(dilation_limit):
                stack.append(
                    ResidualLayer(res_channels, skip_channels, dilation=dilation)
                )

        return stack


class ResidualLayer(torch.nn.Module):
    """Single residual layer

    residual layer is based off of Figure 4 in: https://arxiv.org/abs/1609.03499
                ^
                |
    4)  -------[+]
        |       |
    3)  |    [1 x 1]------> skip-connection
        |       |
    2)  |   ---[x]---
        |   |       |
        | [tanh]   [σ]
        |   |       |
    1)  | [dilated conv]
        |       |
        |------ |
                |
    """

    def __init__(self, res_channels: int, skip_channels: int, dilation: int):
        """Initialize a residual layer.

        Args:
            res_channels: number of channels for i/o in residual layers
            skip_channels: number of channels for skip-connection
            dilation: size of dilation

        """
        super().__init__()
        self.dilated_causal_conv = DilatedCausalConvolution(
            res_channels, res_channels, dilation
        )
        self.gated_activation = GatedActivationUnit()
        self.conv_res = nn.Conv1d(
            res_channels, res_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv_skip = nn.Conv1d(
            res_channels, skip_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, data):
        """Pipeline for a single residual layer"""
        # 1) dilated convolution
        filter_output, gate_output = self.dilated_causal_conv(data)

        # 2) gated activation unit from PixelCNN
        activation_output = self.gated_activation(filter_output, gate_output)

        # 3) 1x1 convs for residual and skip convs respectively
        output_conv_res = self.conv_res(activation_output)
        output_skip = self.conv_skip(activation_output)

        # 4) Residual connection
        output_res = data + output_conv_res

        return output_res, output_skip


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
