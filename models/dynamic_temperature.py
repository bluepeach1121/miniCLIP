import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicTemperature(nn.Module):
    """
    Learns a temperature parameter for the contrastive loss based on
    the variance of image-text similarities in a batch.

    The module:
      1. Takes a single scalar input (the variance).
      2. Feeds it into a small 2-layer MLP.
      3. Outputs log_temp, which is exponentiated to become temperature.
      4. Temperature is then multiplied with the contrastive logits.

    By working in log-space, we ensure temperature is always positive.
    """
    def __init__(self, hidden_dim=16):
        """
        Args:
            hidden_dim (int): Size of the hidden layer in the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dynamic temperature.
        
        Args:
            sim_matrix (torch.Tensor): The NxN image-text similarity matrix
                                       or a subset (e.g., only matching pairs).
        
        Returns:
            temperature (torch.Tensor): A single scalar >= 0, controlling
                                        scaling in the contrastive loss.
        """

        variance = sim_matrix.var() 

        variance_input = variance.unsqueeze(0).unsqueeze(1)  

        # pass through the MLP to get log_temp
        log_temp = self.mlp(variance_input)  

        # exponentiate to ensure positivity
        temperature = torch.exp(log_temp).squeeze(0).squeeze(0) 

        return temperature
