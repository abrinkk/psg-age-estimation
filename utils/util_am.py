import torch
import torch.nn as nn

class am_model(nn.Module):
  def __init__(self, net, in_size):
        """Activation maximization module for networks

        Args:
            net (nn.Module): Network to perform AM on
            in_size (list): List of data input size
        """
        super(am_model, self).__init__()
        self.am_data = nn.Parameter(torch.randn(in_size))
        self.net = net
        for param in self.net.parameters():
          param.requires_grad = False

  def forward(self):
    """Forward call for AM module

    Returns:
        out: Network output
    """
    x = self.net(self.am_data)
    return x