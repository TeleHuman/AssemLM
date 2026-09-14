"""Base module for the AssemLM 2.0 framework."""

from torch import nn


class baseframework(nn.Module):
    """Plain PyTorch base class for the single published framework."""

    def __init__(self):
        super().__init__()
