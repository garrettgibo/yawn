"""Convolution layers for WaveNet"""
import torch
import torch.nn as nn
import wavenet
from wavenet.modules import GatedActivationUnit


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
            out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
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
