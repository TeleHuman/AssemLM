import torch
import torch.nn as nn
from typing import Optional

class PointCloudProjector(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        self.projection_hidden_layer = config.projection_hidden_layer
        self.backbone_output_dim = config.backbone_output_dim
        self.projection_hidden_dim = config.projection_hidden_dim or []
        self.project_output_dim = config.project_output_dim

        # build projector structure
        if self.projection_hidden_layer > 0:
            layers = []
            last_dim = self.backbone_output_dim
            hidden_dims = self.projection_hidden_dim

            for i in range(self.projection_hidden_layer):
                layers.append(nn.LayerNorm(last_dim))
                layers.append(nn.Linear(last_dim, hidden_dims[i]))
                layers.append(nn.GELU())
                last_dim = hidden_dims[i]

            # final layer
            layers.append(nn.Linear(last_dim, self.project_output_dim))
            self.layers = nn.Sequential(*layers)

        else:
            # only one layer
            self.layers = nn.Linear(
                self.backbone_output_dim,
                self.project_output_dim
            )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)
