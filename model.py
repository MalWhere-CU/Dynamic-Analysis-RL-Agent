"""
DDDQN (Double Dueling Deep Q-Network) for Malware Classification.

Source: train/train.ipynb, Cell 8 (lines 1146-1168)
Status: Rewritten to match new checkpoint layer naming convention.
Why: The trained model (models/dqn_cape_best.pth) uses nn.Sequential blocks
     named feature_layer, value_stream, advantage_stream. The computation is
     identical to the original notebook architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DDDQN(nn.Module):
    def __init__(self, dim_states: int, dim_actions: int, dim_hidden: int = 256):
        super(DDDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(dim_states, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dropout = nn.Dropout(0.1)
        self.value_stream = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, dim_actions),
        )

    def forward(self, x: torch.Tensor, avilable_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.feature_layer(x)
        f = self.dropout(f)
        V = self.value_stream(f)
        A = self.advantage_stream(f)
        Q = V + A - torch.mean(A, dim=1, keepdim=True)
        if avilable_actions is not None:
            Q = Q.masked_fill(avilable_actions == 0, -1e9)
        return Q
