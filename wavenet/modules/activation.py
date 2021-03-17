"""Gated activation function"""
import torch
import torch.nn as nn


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
