import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionMLP(nn.Module):
    """
    A two-layer feedforward network that projects an embedding vector
    to the same dimension, applying a skip connection and LayerNorm.
    
    Architecture:
      - Input: [batch_size, embed_dim]
      - Linear(in=embed_dim, out=4*embed_dim)
      - GELU
      - Linear(in=4*embed_dim, out=embed_dim)
      - Residual connection (skip)
      - LayerNorm
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Projection MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, embed_dim].
            
        Returns:
            torch.Tensor: Projected tensor of shape [batch_size, embed_dim].
        """
        residual = x

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        x = x + residual

        x = self.norm(x)

        return x
