"""
Shared fixtures and test utilities for pytorch-module-monitor tests.
"""
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def device():
    """Returns the device to use for tests (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_linear_model():
    """
    Simple 2-layer MLP for basic tests.

    Architecture:
    - Linear(4, 8)
    - ReLU
    - Linear(8, 2)
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    return SimpleModel()


@pytest.fixture
def simple_linear_model_no_bias():
    """
    Simple 2-layer MLP without bias for easier manual computation.

    Architecture:
    - Linear(4, 8, bias=False)
    - ReLU
    - Linear(8, 2, bias=False)
    """
    class SimpleModelNoBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 2, bias=False)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    return SimpleModelNoBias()


@pytest.fixture
def conv_model():
    """
    Simple CNN for testing convolutional layers.

    Architecture:
    - Conv2d(3, 16, kernel_size=3, padding=1)
    - ReLU
    - MaxPool2d(2)
    - Conv2d(16, 32, kernel_size=3, padding=1)
    - ReLU
    - AdaptiveAvgPool2d(1)
    - Flatten
    - Linear(32, 10)
    """
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    return ConvModel()


@pytest.fixture
def sample_batch(device):
    """Small batch of random tensors for testing."""
    batch_size = 4
    input_dim = 4
    return torch.randn(batch_size, input_dim, device=device)


@pytest.fixture
def sample_image_batch(device):
    """Small batch of random images for testing CNNs."""
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    return torch.randn(batch_size, channels, height, width, device=device)


@pytest.fixture
def basic_monitor():
    """Pre-configured ModuleMonitor with default settings."""
    from torch_module_monitor.monitor import ModuleMonitor
    import logging

    logger = logging.getLogger("test_monitor")
    logger.setLevel(logging.DEBUG)

    # Monitor every step for testing
    monitor = ModuleMonitor(
        monitor_step_fn=lambda step: True,
        logger=logger,
        cpu_offload=False
    )
    return monitor


@pytest.fixture
def monitor_with_common_metrics():
    """Monitor with common metrics registered."""
    from torch_module_monitor.monitor import ModuleMonitor
    import logging

    logger = logging.getLogger("test_monitor")
    logger.setLevel(logging.DEBUG)

    # Monitor every step for testing
    monitor = ModuleMonitor(
        monitor_step_fn=lambda step: True,
        logger=logger,
        cpu_offload=False
    )

    # Add common activation metrics
    monitor.add_activation_metric("mean", lambda x: x.mean())
    monitor.add_activation_metric("std", lambda x: x.std())
    monitor.add_activation_metric("abs_max", lambda x: x.abs().max())

    # Add common parameter metrics
    monitor.add_parameter_metric("mean", lambda x: x.mean())
    monitor.add_parameter_metric("std", lambda x: x.std())
    monitor.add_parameter_metric("norm", lambda x: x.norm())

    # Add common gradient metrics
    monitor.add_gradient_metric("mean", lambda x: x.mean())
    monitor.add_gradient_metric("std", lambda x: x.std())
    monitor.add_gradient_metric("norm", lambda x: x.norm())

    return monitor
