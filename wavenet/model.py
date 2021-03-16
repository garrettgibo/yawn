"""
Networks and components that define WaveNet
"""
import torch
import torch.nn as nn


class WaveNet(torch.nn.Module):
    """TODO: class docstring"""

    def __init__(self):
        super().__init__()

    def forward(self, data):
        """TODO: forward pass docstring"""
        return data


class DilatedCauasalConvolution(torch.nn.Module):
    """Creates a causal dilated convolution layer"""

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
    """Creates a gated activation unit layer"""

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


class ResidualBlock(torch.nn.Module):
    """TODO: class docstring"""

    def __init__(self):
        """TODO: forward pass docstring"""
        super().__init__()

    def forward(self, data):
        """TODO: forward pass docstring"""
        return data
