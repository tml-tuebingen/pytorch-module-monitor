

from torchmetrics import Metric
import torch


#################################################################
# Different Metrics that we can use to monitor activations,
# parameters, and gradients.
#
# Activation tensors will be passed in full shape, while
# parameters and gradients will be passed as flattened tensors.
#
# To work with both activation tensors and flattened tensors,
# the metrics should be computed along the last
# dimension of the tensor.
#################################################################
def l1_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L1 norm along last dimension."""
    return torch.linalg.vector_norm(tensor, ord=1, dim=-1)

def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm along last dimension."""
    return torch.linalg.vector_norm(tensor, ord=2, dim=-1)

def mean(tensor: torch.Tensor) -> torch.Tensor:
    """Compute mean along last dimension."""
    return torch.mean(tensor, dim=-1)

def std(tensor: torch.Tensor) -> torch.Tensor:
    """Compute standard deviation along last dimension."""
    return torch.std(tensor, dim=-1)

def max_value(tensor: torch.Tensor) -> torch.Tensor:
    """Compute maximum value along last dimension."""
    return torch.max(tensor, dim=-1).values

def min_value(tensor: torch.Tensor) -> torch.Tensor:
    """Compute minimum value along last dimension."""
    return torch.min(tensor, dim=-1).values

def sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Compute sparsity (fraction of near-zero values) along last dimension."""
    zeros = (torch.abs(tensor) < threshold).float()
    return torch.mean(zeros, dim=-1)
