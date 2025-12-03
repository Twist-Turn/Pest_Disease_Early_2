"""
Very simplified DreamerV3-style policy/value model.

In real DreamerV3 you would learn a latent dynamics model and plan in latent space.
Here we only provide a feedforward policy/value head on top of the encoded state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterventionPolicy(nn.Module):
    """
    Outputs timing recommendations for pest/disease intervention.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.gelu(self.fc1(state))
        x = F.gelu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
